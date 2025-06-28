#!/usr/bin/env python
"""
测试训练过程中的写入条件，分析为什么写入较少
"""

import torch
import logging
from ntm.long_term_memory import Neo4jGraphMemory
from ntm.encoder_decoder import encode_text_to_vector, decode_add_vector_to_text
import os
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_write_conditions():
    """测试训练过程中的写入条件"""
    
    # 创建Neo4j记忆实例
    memory = Neo4jGraphMemory(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password123"),
        node_dim=8,
        encoder=encode_text_to_vector,
        decoder=decode_add_vector_to_text,
        log_write=True
    )
    
    logger.info("开始测试训练写入条件...")
    
    # 模拟训练过程中的不同attention权重模式
    test_cases = [
        # 1. 均匀分布的attention权重（可能不会写入）
        {
            "name": "均匀attention权重",
            "w_t": torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]]),
            "w_prev": torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]]),
            "e": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]),
            "a": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        },
        # 2. 有明确峰值的attention权重（应该会写入）
        {
            "name": "峰值attention权重",
            "w_t": torch.tensor([[0.8, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]]),
            "w_prev": torch.tensor([[0.1, 0.8, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]]),
            "e": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]),
            "a": torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
        },
        # 3. 随机attention权重
        {
            "name": "随机attention权重",
            "w_t": torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]]),
            "w_prev": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]]),
            "e": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]),
            "a": torch.tensor([[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        }
    ]
    
    for i, case in enumerate(test_cases):
        logger.info(f"\n测试案例 {i+1}: {case['name']}")
        logger.info(f"w_t: {case['w_t'][0].tolist()}")
        logger.info(f"w_t最大值索引: {case['w_t'][0].argmax().item()}")
        logger.info(f"w_t最大值: {case['w_t'][0].max().item():.3f}")
        
        # 执行写入
        memory.write(case['w_t'], case['w_prev'], case['e'], case['a'])
    
    # 获取写入日志
    log = memory.get_write_log()
    logger.info(f"\n总共写入 {len(log)} 条记录")
    
    # 导出图数据
    graph_data = memory.export_entailment_graph_json()
    logger.info(f"图数据节点数: {len(graph_data.get('nodes', []))}")
    logger.info(f"图数据边数: {len(graph_data.get('edges', []))}")
    
    # 显示写入日志
    for i, entry in enumerate(log):
        logger.info(f"日志 {i+1}: source_idx={entry['source_idx']}, proposition={entry['proposition']}")
    
    memory.close()
    return len(log)

def simulate_training_sequence():
    """模拟训练序列的写入情况"""
    
    memory = Neo4jGraphMemory(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password123"),
        node_dim=8,
        encoder=encode_text_to_vector,
        decoder=decode_add_vector_to_text,
        log_write=True
    )
    
    logger.info("\n模拟训练序列写入...")
    
    # 模拟一个训练序列（10个时间步）
    for t in range(10):
        # 模拟训练过程中attention权重的变化
        if t < 5:
            # 前5步：写入阶段，attention权重有峰值
            w_t = torch.tensor([[0.7, 0.2, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]])
        else:
            # 后5步：读取阶段，attention权重可能更均匀
            w_t = torch.tensor([[0.3, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]])
        
        w_prev = torch.tensor([[0.2, 0.3, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0]])
        e = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        a = torch.tensor([[0.1 + t*0.1, 0.2 + t*0.1, 0.3 + t*0.1, 0.4 + t*0.1, 
                          0.5 + t*0.1, 0.6 + t*0.1, 0.7 + t*0.1, 0.8 + t*0.1]])
        
        logger.info(f"时间步 {t+1}: w_t最大值={w_t[0].max().item():.3f}, 索引={w_t[0].argmax().item()}")
        memory.write(w_t, w_prev, e, a)
    
    log = memory.get_write_log()
    logger.info(f"序列写入记录数: {len(log)}")
    
    memory.close()

if __name__ == "__main__":
    test_training_write_conditions()
    simulate_training_sequence() 