"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from .ntm import NTM
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.输入向量的维度（外部输入大小）
        :param num_outputs: External number of outputs. 输出向量的维度（外部输出大小）
        :param controller_size: The size of the internal representation.控制器（LSTM）隐藏状态的维度
        :param controller_layers: Controller number of layers.LSTM控制器的层数
        :param num_heads: Number of heads. 读写头的数量
        :param N: Number of rows in the memory bank.记忆矩阵行数（记忆单元数）
        :param M: Number of cols/features in the memory bank.记忆矩阵每个单元的特征维度（列数）
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args  保存参数方便调用
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M

        # Create the NTM components
        memory = NTMMemory(N, M)
        controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
        heads = nn.ModuleList([])
        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, controller_size),
                NTMWriteHead(memory, controller_size)
            ]

        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads) # 这行组装了整个 NTM，我们的主角就是 self.ntm(x, prev_state)，每一个时间步调用一次。
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs) # shape: [B, I]

        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
