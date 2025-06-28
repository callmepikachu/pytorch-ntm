"""
Dual-Memory Copy Task NTM model.

Implements the copy task using a Dual-Memory NTM that includes both:
- Short-term memory (matrix-based)
- Long-term memory (graph-based GNN)
"""

import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

# 假设你的 DualMemoryNTM 已经在 ntm.aio 中定义好
from ntm.aio import EncapsulatedDualMemoryNTM  # 替换为你实际的类名


# Generator of randomized test sequences
def dual_copy_dataloader(num_batches,
                         batch_size,
                         seq_width,
                         min_len,
                         max_len):
    """
    生成用于 Dual-Memory NTM 训练的随机二进制序列数据。
    Generator of random sequences for the dual-memory copy task.

    Creates random batches of binary sequences.

    All sequences within each batch have the same length.
    The length is in range [min_len, max_len]

    :param num_batches: Total number of batches to generate.
    :param batch_size: Number of samples per batch.
    :param seq_width: Width of each item in sequence.
    :param min_len: Minimum sequence length.
    :param max_len: Maximum sequence length.
    num_batches: 要生成的总 batch 数量
    batch_size: 每个 batch 的样本数
    seq_width: 序列中每个元素的宽度（bit 数）
    min_len, max_len: 每个序列长度的最小和最大值
     📌 输出：

    每次 yield 一个 batch，格式为 (batch_num, input_seq, output_seq)，其中：

    input_seq: [seq_len+1 x batch_size x seq_width+1]，包含分隔符（delimiter）。
    output_seq: [seq_len x batch_size x seq_width]，与输入相同但不含 delimiter。
    """
    for batch_num in range(num_batches):

        # All sequences in the same batch have the same length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq).float()

        # Input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0  # Delimiter in control channel
        outp = seq.clone()

        yield batch_num + 1, inp, outp


@attrs
class DualCopyTaskParams(object):
    """
    Parameters class for the Dual-Memory Copy Task.

    Used to define all hyperparameters required by the model and training process.

    Fields:
        name: Task name
        controller_size: Controller hidden size
        controller_layers: Number of controller layers
        num_heads: Number of read/write heads
        sequence_width: Binary vector width of each sequence element
        sequence_min_len/max_len: Length range of generated sequences
        short_term_n/m: Short-term memory matrix size (rows/columns)
        long_term_nodes/dim: Long-term memory graph node count and feature dimension
        num_batches: Total number of batches to train
        batch_size: Size of each batch
        rmsprop_lr/momentum/alpha: Optimizer parameters for RMSProp
    """
    name = attrib(default="dual-copy-task")
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)
    num_heads = attrib(default=1, converter=int)

    sequence_width = attrib(default=8, converter=int)
    sequence_min_len = attrib(default=1, converter=int)
    sequence_max_len = attrib(default=20, converter=int)

    short_term_n = attrib(default=128, converter=int)   # Memory rows
    short_term_m = attrib(default=20, converter=int)    # Memory cols
    long_term_nodes = attrib(default=32, converter=int)  # Graph nodes
    long_term_dim = attrib(default=8, converter=int)     # Node feature dimension

    num_batches = attrib(default=50000, converter=int)
    batch_size = attrib(default=1, converter=int)

    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)


@attrs
class DualCopyTaskModelTraining:
    params = attrib(default=Factory(DualCopyTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()
    long_term_memory_backend = attrib(default="in-memory")
    neo4j_config = attrib(default=None)

    @net.default
    def default_net(self):
        # 确保long_term_memory_backend有默认值
        backend = getattr(self, 'long_term_memory_backend', 'in-memory')
        
        # 从环境变量获取Neo4j配置（如果未提供）
        if backend == "neo4j" and getattr(self, 'neo4j_config', None) is None:
            import os
            self.neo4j_config = {
                "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "user": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password123")
            }
        
        return EncapsulatedDualMemoryNTM(
            input_size=self.params.sequence_width + 1,
            output_size=self.params.sequence_width,
            controller_size=self.params.controller_size,
            controller_layers=self.params.controller_layers,
            num_heads=self.params.num_heads,
            short_term_memory=(self.params.short_term_n, self.params.short_term_m),
            long_term_memory=(self.params.long_term_nodes, self.params.long_term_dim),
            long_term_memory_backend=backend,
            neo4j_config=getattr(self, 'neo4j_config', None)
        )

    @dataloader.default
    def default_dataloader(self):
        return dual_copy_dataloader(
            self.params.num_batches,
            self.params.batch_size,
            self.params.sequence_width,
            self.params.sequence_min_len,
            self.params.sequence_max_len
        )

    @criterion.default
    def default_criterion(self):
        # 使用 BCEWithLogitsLoss 替代 BCELoss
        # 这个损失函数会自动应用 sigmoid，所以控制器不需要输出 sigmoid
        return nn.BCEWithLogitsLoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(
            self.net.parameters(),
            lr=self.params.rmsprop_lr,
            momentum=self.params.rmsprop_momentum,
            alpha=self.params.rmsprop_alpha
        )