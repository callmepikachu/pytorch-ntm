import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np
from .encoder_decoder import encode_text_to_vector, decode_add_vector_to_text
import json


class AbstractGraphMemory(nn.Module, ABC):
    """图记忆后端接口，定义 read、write、reset 等操作。"""
    @abstractmethod
    def write(self, w_t, w_prev, e, a):
        pass

    @abstractmethod
    def read(self, key_vector):
        pass

    @abstractmethod
    def reset(self, batch_size=1):
        pass


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


class InMemoryGraphMemory(AbstractGraphMemory):
    def __init__(self, num_nodes, node_dim):
        super(InMemoryGraphMemory, self).__init__()
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


class Neo4jGraphMemory(AbstractGraphMemory):
    """
    使用 Neo4j 作为后端的图记忆实现。
    需安装 neo4j、numpy、transformers 等依赖。
    """
    def __init__(self, uri, user, password, node_dim, encoder=None, decoder=None, log_write=False):
        super(Neo4jGraphMemory, self).__init__()
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.node_dim = node_dim
        self.encoder = encoder if encoder is not None else encode_text_to_vector
        self.decoder = decoder if decoder is not None else decode_add_vector_to_text
        self.log_write = log_write
        self._log = []

    def close(self):
        self.driver.close()

    def write(self, w_t, w_prev, e, a):
        # 仅处理 batch 的第一个样本
        a_vec = a[0].detach().cpu().numpy()
        # 解码向量为命题文本
        proposition = self.decoder(a_vec) if self.decoder else str(a_vec.tolist())
        with self.driver.session() as session:
            # 查找 attention 最大的节点作为 source
            source_idx = int(w_t[0].argmax().item())
            source_node = session.run(
                "MATCH (n:Proposition {idx: $idx}) RETURN n", {"idx": source_idx}
            ).single()
            # 若不存在则新建 source
            if not source_node:
                session.run(
                    "CREATE (n:Proposition {idx: $idx, text: $text})",
                    {"idx": source_idx, "text": f"Node_{source_idx}"}
                )
            # 新建命题节点
            session.run(
                "CREATE (m:Proposition {text: $text, embedding: $embedding})",
                {"text": proposition, "embedding": a_vec.tolist()}
            )
            # 建立 ENTAILS 关系
            session.run(
                "MATCH (n:Proposition {idx: $idx}), (m:Proposition {text: $text}) "
                "MERGE (n)-[:ENTAILS]->(m)",
                {"idx": source_idx, "text": proposition}
            )
            if self.log_write:
                self._log.append({
                    "source_idx": source_idx,
                    "proposition": proposition,
                    "embedding": a_vec.tolist()
                })

    def read(self, key):
        # 仅处理 batch 的第一个样本
        key_vec = key[0].detach().cpu().numpy()
        with self.driver.session() as session:
            # 查找 embedding 最相近的命题节点
            result = session.run(
                "MATCH (n:Proposition) WHERE n.embedding IS NOT NULL "
                "RETURN n.text AS text, n.embedding AS embedding"
            )
            min_dist = float('inf')
            best_text = None
            best_emb = None
            for record in result:
                emb = record["embedding"]
                text = record["text"]
                dist = np.linalg.norm(key_vec - np.array(emb))
                if dist < min_dist:
                    min_dist = dist
                    best_text = text
                    best_emb = emb
            # 判空保护
            if best_emb is None:
                best_emb = np.zeros_like(key_vec)
                best_text = ""
            # 获取邻接节点
            forward = session.run(
                "MATCH (n:Proposition {text: $text})-[:ENTAILS]->(m) RETURN m.embedding AS embedding",
                {"text": best_text}
            )
            backward = session.run(
                "MATCH (m:Proposition)-[:ENTAILS]->(n:Proposition {text: $text}) RETURN m.embedding AS embedding",
                {"text": best_text}
            )
            # 聚合邻居 embedding
            f_embs = [np.array(r["embedding"]) for r in forward if r["embedding"] is not None]
            b_embs = [np.array(r["embedding"]) for r in backward if r["embedding"] is not None]
            # 若无邻居则用自身
            r_normal = np.array(best_emb)
            r_forward = np.mean(f_embs, axis=0) if f_embs else r_normal
            r_backward = np.mean(b_embs, axis=0) if b_embs else r_normal
            # 转为 torch tensor
            r_normal = torch.tensor(r_normal, dtype=torch.float).unsqueeze(0)
            r_forward = torch.tensor(r_forward, dtype=torch.float).unsqueeze(0)
            r_backward = torch.tensor(r_backward, dtype=torch.float).unsqueeze(0)
            return r_normal, r_forward, r_backward

    def reset(self, batch_size=1):
        # 可选：清空数据库或跳过
        pass

    def get_write_log(self):
        return self._log

    def export_entailment_graph_json(self, file_path=None):
        """
        导出所有命题节点及 ENTAILS 关系为 JSON 格式。
        {"nodes": [...], "edges": [...]}
        """
        with self.driver.session() as session:
            nodes = []
            node_map = {}
            result = session.run(
                "MATCH (n:Proposition) RETURN n.idx AS idx, n.text AS text, n.embedding AS embedding"
            )
            for rec in result:
                node_id = str(rec["idx"]) if rec["idx"] is not None else rec["text"]
                node_map[rec["text"]] = node_id
                nodes.append({
                    "id": node_id,
                    "text": rec["text"],
                    "embedding": rec["embedding"]
                })
            edges = []
            result = session.run(
                "MATCH (n:Proposition)-[:ENTAILS]->(m:Proposition) RETURN n.text AS source, m.text AS target"
            )
            for rec in result:
                src = node_map.get(rec["source"], rec["source"])
                tgt = node_map.get(rec["target"], rec["target"])
                edges.append({"source": src, "target": tgt})
            graph = {"nodes": nodes, "edges": edges}
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(graph, f, ensure_ascii=False, indent=2)
            return graph
