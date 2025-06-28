#!/usr/bin/env python
import torch
from torch import nn
import torch.nn.functional as F

from ntm.dual_controller import DualMemoryController
from ntm.memory import NTMMemory
from ntm.long_term_memory import InMemoryGraphMemory, Neo4jGraphMemory


class NTM(nn.Module):
    """A Neural Turing Machine."""
    def __init__(self, num_inputs, num_outputs, controller, memory, heads):
        """Initialize the NTM.

        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`NTMMemory`
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`

        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(NTM, self).__init__()

        # Save arguments
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads

        self.N, self.M = memory.size()
        _, self.controller_size = controller.size()

        # Initialize the initial previous read values to random biases
        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, num_outputs)
        self.reset_parameters()

    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        return init_r, controller_state, heads_state

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, x, prev_state):
        """NTM forward function.

        :param x: input vector (batch_size x num_inputs)    x: [B, I] 输入向量
        :param prev_state: The previous state of the NTM, 包含:
            prev_reads: List[Tensor[B, M]]，每个读头的输出
            prev_controller_state: LSTM 的隐状态
            prev_heads_states: 每个读写头的状态（比如地址权重等）
        """
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        inp = torch.cat([x] + prev_reads, dim=1) # x.shape: [B, I], prev_reads: 一般是 1 个读头，所以 [B, M]; inp: [B, I + R*M],R=1
        controller_outp, controller_state = self.controller(inp, prev_controller_state)#         controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers) [B, I + M] --> [B, C], C 是 controller_size

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(controller_outp, prev_head_state)
                reads += [r]
            else:
                head_state = head(controller_outp, prev_head_state)
            heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = F.sigmoid(self.fc(inp2))

        # Pack the current state
        state = (reads, controller_state, heads_states)

        return o, state


class DualMemoryNTM(nn.Module):
    def __init__(self, input_size, output_size, controller_size,
                 short_term_memory, long_term_memory,
                 long_term_memory_backend="in-memory",
                 neo4j_config=None,
                 encoder=None, decoder=None):
        """
        Args:
            input_size: int - 输入维度
            output_size: int - 输出维度
            controller_size: int - 控制器隐藏层大小
            short_term_memory: tuple(N, M) - 短期记忆矩阵大小
            long_term_memory: tuple(num_nodes, node_dim) - 长期记忆图结构大小
            long_term_memory_backend: str - "in-memory" 或 "neo4j"
            neo4j_config: dict - 连接 Neo4j 所需参数（uri, user, password）
            encoder/decoder: 命题向量与文本的互转函数
        """
        super(DualMemoryNTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size

        st_n, st_m = short_term_memory
        lt_nodes, lt_dim = long_term_memory

        self.short_term = NTMMemory(st_n, st_m)

        if long_term_memory_backend == "in-memory":
            self.long_term = InMemoryGraphMemory(lt_nodes, lt_dim)
        elif long_term_memory_backend == "neo4j":
            if neo4j_config is None:
                raise ValueError("neo4j_config 必须提供 uri, user, password")
            self.long_term = Neo4jGraphMemory(
                uri=neo4j_config["uri"],
                user=neo4j_config["user"],
                password=neo4j_config["password"],
                node_dim=lt_dim,
                encoder=encoder,
                decoder=decoder
            )
        else:
            raise ValueError(f"未知 long_term_memory_backend: {long_term_memory_backend}")

        self.controller = DualMemoryController(
            input_size=input_size,
            hidden_size=controller_size,
            output_size=output_size,
            memory_size=st_n,
            memory_dim=st_m,
            long_term_nodes=lt_nodes,
            long_term_dim=lt_dim
        )
        self._prev_w_lt = None

    def forward(self, input_seq):
        """
        前向传播

        Args:
            input_seq: 输入序列
                - 如果是3D tensor: [seq_len, batch_size, input_size]
                - 如果是2D tensor: [batch_size, input_size] (单个时间步)
                - 如果是1D tensor: [input_size] (单个样本，单个时间步)

        Returns:
            outputs: 输出序列 [seq_len, batch_size, output_size]
        """
        # Handle different input dimensions
        if input_seq.dim() == 1:
            # Single sample, single timestep: [input_size] -> [1, 1, input_size]
            input_seq = input_seq.unsqueeze(0).unsqueeze(0)
            batch_size = 1
            seq_len = 1
        elif input_seq.dim() == 2:
            # Single timestep: [batch_size, input_size] -> [1, batch_size, input_size]
            input_seq = input_seq.unsqueeze(0)
            batch_size = input_seq.size(1)
            seq_len = 1
        else:
            # Full sequence: [seq_len, batch_size, input_size]
            seq_len, batch_size, _ = input_seq.shape

        # 确保控制器状态被正确初始化
        if self.controller.hidden is None or self.controller.hidden.size(0) != batch_size:
            self.controller.reset(batch_size)

        outputs = []

        # Process each timestep
        for t in range(seq_len):
            x = input_seq[t]  # [batch_size, input_size]

            # Generate keys for memory addressing
            key_st = self.controller.generate_key_st(x)
            key_lt = self.controller.generate_key_lt(x)

            # Short-term memory read using content addressing
            memory_reshaped = self.short_term.memory  # [batch_size, N, M]
            key_expanded = key_st.unsqueeze(1)  # [batch_size, 1, M]

            # Compute cosine similarity
            similarity = torch.cosine_similarity(key_expanded, memory_reshaped, dim=2)  # [batch_size, N]
            w_st_read = torch.softmax(similarity, dim=1)  # [batch_size, N]

            # Read from short-term memory
            short_term_read = self.short_term.read(w_st_read)

            # 长期记忆读取
            ltm_normal, ltm_forward, ltm_backward = self.long_term.read(key_lt)

            # 控制器前馈
            output = self.controller(x, short_term_read, ltm_normal, ltm_forward, ltm_backward)
            outputs.append(output)

            # 写入操作
            w_st = self.controller.generate_write_weights_st()
            e_st = self.controller.generate_erase_st()
            a_st = self.controller.generate_add_st()
            self.short_term.write(w_st, e_st, a_st)

            # Write to long-term memory
            w_lt = self.controller.generate_write_weights_lt()
            e_lt = self.controller.generate_erase_lt()
            a_lt = self.controller.generate_add_lt()

            # Use previous weights for long-term memory update
            if self._prev_w_lt is not None:
                self.long_term.write(w_lt, self._prev_w_lt, e_lt, a_lt)
            else:
                # First timestep: use zeros as previous weights
                prev_w_lt = torch.zeros_like(w_lt)
                self.long_term.write(w_lt, prev_w_lt, e_lt, a_lt)

            self._prev_w_lt = w_lt.detach()

        # Stack outputs: [seq_len, batch_size, output_size]
        return torch.stack(outputs)

    def init_sequence(self, batch_size):
        """初始化短期和长期记忆"""
        self.short_term.reset(batch_size)
        self.long_term.reset(batch_size)
        self._prev_w_lt = None
        self.controller.reset(batch_size)

    def calculate_num_params(self):
        """计算总参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)