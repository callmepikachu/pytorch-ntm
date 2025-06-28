import pytest
import torch
from ntm.memory import NTMMemory
from ntm.long_term_memory import InMemoryGraphMemory, Neo4jGraphMemory
from ntm.encoder_decoder import encode_text_to_vector, decode_add_vector_to_text
import numpy as np

def _t(*l):
    return torch.Tensor(l).unsqueeze(0)

class TestMemoryReadWrite:
    N = 4
    M = 4

    def setup_class(self):
        self.memory = NTMMemory(self.N, self.M)
        self.memory.reset(batch_size=1)

    def teardown_class(self):
        del self.memory

    def test_size(self):
        n, m = self.memory.size()
        assert n == self.N
        assert m == self.M

    @pytest.mark.parametrize('w, e, a, expected', [
        (_t(1, 0, 0, 0), _t(1, 1, 1, 1), _t(1, 0, 0, 0), _t(1, 0, 0, 0)),
        (_t(0, 1, 0, 0), _t(1, 1, 1, 1), _t(0, 1, 0, 0), _t(0, 1, 0, 0)),
        (_t(0, 0, 1, 0), _t(1, 1, 1, 1), _t(0, 0, 1, 0), _t(0, 0, 1, 0)),
        (_t(0, 0, 0, 1), _t(1, 1, 1, 1), _t(0, 0, 0, 1), _t(0, 0, 0, 1)),
        (_t(1, 0, 0, 0), _t(0, 1, 1, 1), _t(0, 1, 1, 1), _t(1, 1, 1, 1)),
        (_t(0, 1, 0, 0), _t(0, 0, 0, 0), _t(0, 0, 1, 0), _t(0, 1, 1, 0)),
        (_t(0, 0, 1, 0), _t(0, 0, 0, 0), _t(0, 0, 0, 0), _t(0, 0, 1, 0)),
        (_t(0, 0, 0, 1), _t(0, 0, 0, 0.5), _t(0, 0, 0, 0.2), _t(0, 0, 0, 0.7)),
        (_t(0.5, 0.5, 0, 0), _t(1, 1, 1, 1), _t(0, 0, 0, 0), _t(0.25, 0.5, 0.5, 0.25)),
    ])
    def test_read_write(self, w, e, a, expected):
        self.memory.write(w, e, a)
        result = self.memory.read(w)
        assert torch.equal(expected.data, result.data)


@pytest.fixture
def mem():
    mm = NTMMemory(4, 4)
    mm.reset(batch_size=1)

    # Identity-fy the memory matrix
    mm.write(_t(1, 0, 0, 0), _t(1, 1, 1, 1), _t(1, 0, 0, 0))
    mm.write(_t(0, 1, 0, 0), _t(1, 1, 1, 1), _t(0, 1, 0, 0))
    mm.write(_t(0, 0, 1, 0), _t(1, 1, 1, 1), _t(0, 0, 1, 0))
    mm.write(_t(0, 0, 0, 1), _t(1, 1, 1, 1), _t(0, 0, 0, 1))

    return mm


class TestAddressing:

    @pytest.mark.parametrize('k, beta, g, shift, gamma, w_prev, expected', [
        (_t(1, 0, 0, 0), _t(100), _t(1), _t(0, 1, 0), _t(100), _t(0, 0, 0, 0), _t(1, 0, 0, 0)), # test similarity/interpolation
    ])
    def test_addressing(self, mem, k, beta, g, shift, gamma, w_prev, expected):
        w = mem.address(k, beta, g, shift, gamma, w_prev)
        assert torch.equal(w.data, expected.data)

def test_inmemory_graph_memory_update_E_matrix():
    ltm = InMemoryGraphMemory(num_nodes=4, node_dim=2)
    E_initial = ltm.E.clone().detach()
    w_t = torch.tensor([[1., 0., 0., 0.]])
    w_prev = torch.tensor([[0., 1., 0., 0.]])
    e = torch.rand(1, 2)
    a = torch.rand(1, 2)
    expected_E = E_initial + torch.outer(w_t[0], w_prev[0])
    ltm.write(w_t, w_prev, e, a)
    assert torch.allclose(ltm.E.data, expected_E), "E matrix update failed!"

def test_inmemory_graph_memory_forward_backward_read():
    ltm = InMemoryGraphMemory(num_nodes=3, node_dim=2)
    with torch.no_grad():
        ltm.M.data = torch.tensor([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        ltm.E.data = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ])
    key = torch.tensor([[1.0, 0.0]])
    r_normal, r_forward, r_backward = ltm.read(key)
    assert r_normal.shape == (1, 2)
    assert r_forward.shape == (1, 2)
    assert r_backward.shape == (1, 2)

def test_encode_decode_text_vector():
    text = "The cat sat on the mat."
    vec = encode_text_to_vector(text)
    assert isinstance(vec, (np.ndarray, torch.Tensor))
    text2 = decode_add_vector_to_text(vec)
    assert isinstance(text2, str)
    assert len(text2) > 0

# Neo4jGraphMemory 测试（需本地有 Neo4j 实例）
# def test_neo4j_graph_memory_write_read():
#     neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"}
#     ltm = Neo4jGraphMemory(**neo4j_config, node_dim=8)
#     w_t = torch.tensor([[1., 0., 0., 0.]])
#     w_prev = torch.tensor([[0., 1., 0., 0.]])
#     e = torch.rand(1, 8)
#     a = torch.rand(1, 8)
#     ltm.write(w_t, w_prev, e, a)
#     key = torch.rand(1, 8)
#     r_normal, r_forward, r_backward = ltm.read(key)
#     assert r_normal.shape == (1, 8)
#     ltm.close()