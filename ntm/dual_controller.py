import torch
import torch.nn as nn
import torch.nn.functional as F

class DualMemoryController(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, memory_size, memory_dim, long_term_nodes, long_term_dim):
        super(DualMemoryController, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.long_term_nodes = long_term_nodes
        self.long_term_dim = long_term_dim

        # 计算 RNN 输入维度：input + short_term_read + ltm_normal + ltm_forward + ltm_backward
        rnn_input_size = input_size + memory_dim + long_term_dim + long_term_dim + long_term_dim

        # RNN 核心
        self.rnn = nn.LSTMCell(rnn_input_size, hidden_size)

        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)

        # 接口生成层
        self.key_st_generator = nn.Linear(hidden_size, memory_dim)  # 短期记忆键
        self.key_lt_generator = nn.Linear(hidden_size, long_term_dim)  # 长期记忆键

        self.w_st_generator = nn.Linear(hidden_size, memory_size)
        self.e_st_generator = nn.Linear(hidden_size, memory_dim)
        self.a_st_generator = nn.Linear(hidden_size, memory_dim)

        self.w_lt_generator = nn.Linear(hidden_size, long_term_nodes)
        self.e_lt_generator = nn.Linear(hidden_size, long_term_dim)
        self.a_lt_generator = nn.Linear(hidden_size, long_term_dim)

        # 初始状态
        self.hidden = None
        self.cell = None

    def reset(self, batch_size):
        """重置控制器状态"""
        device = next(self.parameters()).device  # 获取模型设备
        self.hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        self.cell = torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, x, short_term_read, ltm_normal, ltm_forward, ltm_backward):
        """
        Args:
            x: 当前输入 [input_size] or [B, input_size]
            short_term_read: 来自短期记忆的读取结果 [B, memory_dim]
            ltm_normal: 正常读取长期记忆 [B, long_term_dim]
            ltm_forward: 前向读取长期记忆 [B, long_term_dim]
            ltm_backward: 后向读取长期记忆 [B, long_term_dim]

        Returns:
            output: 当前控制器输出 [B, output_size]
        """
        # 获取批次大小
        batch_size = short_term_read.shape[0]

        # 确保 x 有正确的维度
        if x.dim() == 1:
            # 如果 x 是一维的，扩展到批次维度
            x = x.unsqueeze(0).expand(batch_size, -1)
        elif x.shape[0] != batch_size:
            # 如果 x 的批次大小不匹配，扩展它
            x = x.expand(batch_size, -1)

        # 确保所有输入都是二维的
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if short_term_read.dim() == 1:
            short_term_read = short_term_read.unsqueeze(0)
        if ltm_normal.dim() == 1:
            ltm_normal = ltm_normal.unsqueeze(0)
        if ltm_forward.dim() == 1:
            ltm_forward = ltm_forward.unsqueeze(0)
        if ltm_backward.dim() == 1:
            ltm_backward = ltm_backward.unsqueeze(0)

        combined_input = torch.cat([x, short_term_read, ltm_normal, ltm_forward, ltm_backward], dim=-1)
        h, c = self.rnn(combined_input, (self.hidden, self.cell) if self.hidden is not None else None)
        self.hidden = h
        self.cell = c
        output = self.output_layer(h)
        return output

    # ---------------------- Short-Term Memory 接口 ----------------------

    def generate_key_st(self, x):
        return torch.tanh(self.key_st_generator(self.hidden))

    def generate_key_lt(self, x):
        return torch.tanh(self.key_lt_generator(self.hidden))

    def generate_write_weights_st(self):
        return F.softmax(self.w_st_generator(self.hidden), dim=-1)

    def generate_erase_st(self):
        return torch.sigmoid(self.e_st_generator(self.hidden))

    def generate_add_st(self):
        return self.a_st_generator(self.hidden)

    # ---------------------- Long-Term Memory 接口 ----------------------

    def generate_write_weights_lt(self):
        return F.softmax(self.w_lt_generator(self.hidden), dim=-1)

    def generate_erase_lt(self):
        return torch.sigmoid(self.e_lt_generator(self.hidden))

    def generate_add_lt(self):
        return self.a_lt_generator(self.hidden)