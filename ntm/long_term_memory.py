import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGNN(nn.Module):
    """简单图神经网络模块"""
    def __init__(self, input_dim, hidden_dim=64):
        super(GraphGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, M, E):
        """
        Args:
            M: [N x D] 记忆矩阵（N个节点，D维）
            E: [N x N] 图关系矩阵
        Returns:
            M_gnn: [N x D] 更新后的记忆内容
        """
        # 使用邻接矩阵聚合邻居信息
        neighbor_agg = torch.matmul(E, M)
        out = F.relu(self.fc1(neighbor_agg))
        M_gnn = self.fc2(out)
        return M_gnn


class LongTermMemory(nn.Module):
    def __init__(self, num_nodes, node_dim):
        super(LongTermMemory, self).__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim

        # 初始化记忆矩阵 M 和关系矩阵 E
        self.M = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.E = nn.Parameter(torch.randn(num_nodes, num_nodes))

        # GNN 模块
        self.gnn = GraphGNN(node_dim)

        # 保存上一步的写入权重用于 E 更新
        self.register_buffer('prev_w', torch.zeros(num_nodes))

    def write(self, w_t, w_prev, e, a):
        """
        Args:
            w_t: 当前 attention 权重 [B, N] -> 修复：处理批次维度
            w_prev: 上一步 attention 权重 [B, N]
            e: erase vector [B, D]
            a: add vector [B, D]
        """
        batch_size = w_t.shape[0]

        # 对每个批次分别处理
        for b in range(batch_size):
            w_t_b = w_t[b]  # [N]
            e_b = e[b]      # [D]
            a_b = a[b]      # [D]

            # Step 1: Erase
            erase_weight = torch.outer(w_t_b, e_b)
            erased_M = self.M * (1 - erase_weight)

            # Step 2: Add
            add_weight = torch.outer(w_t_b, a_b)
            new_M_hat = erased_M + add_weight

            # Step 3: Update E matrix using previous weight
            if w_prev is not None:
                w_prev_b = w_prev[b]
                self.E.data += torch.outer(w_t_b, w_prev_b)

            # Step 4: 直接使用 new_M_hat 更新 memory
            self.M.data = new_M_hat

        # 更新 prev_w（使用第一个批次的权重，或者可以改为平均）
        self.prev_w.data = w_t[0].detach()

    def read(self, key_vector):
        """
        执行读取操作（content-based addressing）

        Args:
            key_vector: [B, D] 查询 key

        Returns:
            r_normal: 正常读取结果 [B, D]
            r_forward: 前向读取 [B, D]
            r_backward: 反向读取 [B, D]
        """
        batch_size = key_vector.shape[0]

        # 批量计算相似度
        sim = F.cosine_similarity(key_vector.unsqueeze(1), self.M.unsqueeze(0), dim=2)  # [B, N]
        beta = 100.0
        gamma = 1.0
        w0 = F.softmax(beta * sim, dim=1)  # [B, N]
        w = F.normalize(w0 ** gamma, p=1, dim=1)  # [B, N]

        # Normal read
        r_normal = torch.matmul(w, self.M)  # [B, N] x [N, D] -> [B, D]

        # Forward read
        r_forward = torch.matmul(w, torch.matmul(self.E, self.M))  # [B, D]

        # Backward read
        r_backward = torch.matmul(w, torch.matmul(self.E.t(), self.M))  # [B, D]

        return r_normal, r_forward, r_backward

    def reset(self, batch_size=1):
        """重置记忆状态"""
        device = self.M.device
        self.prev_w = torch.zeros(self.num_nodes, device=device)
