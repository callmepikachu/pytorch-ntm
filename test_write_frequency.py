#!/usr/bin/env python
"""
测试Neo4j写入频率和条件的脚本
"""

import torch
import logging
from ntm.long_term_memory import Neo4jGraphMemory
from ntm.encoder_decoder import encode_text_to_vector, decode_add_vector_to_text
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_write_frequency():
    """测试Neo4j写入频率"""
    
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
    
    logger.info("开始测试写入频率...")
    
    # 模拟训练过程中的写入操作
    for i in range(10):
        # 模拟不同的attention权重
        w_t = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]])
        w_prev = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]])
        e = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        a = torch.tensor([[0.1 + i*0.1, 0.2 + i*0.1, 0.3 + i*0.1, 0.4 + i*0.1, 
                          0.5 + i*0.1, 0.6 + i*0.1, 0.7 + i*0.1, 0.8 + i*0.1]])
        
        logger.info(f"写入第 {i+1} 次，add vector: {a[0].tolist()}")
        memory.write(w_t, w_prev, e, a)
    
    # 获取写入日志
    log = memory.get_write_log()
    logger.info(f"总共写入 {len(log)} 条记录")
    
    # 导出图数据
    graph_data = memory.export_entailment_graph_json()
    logger.info(f"图数据节点数: {len(graph_data.get('nodes', []))}")
    logger.info(f"图数据边数: {len(graph_data.get('edges', []))}")
    
    # 显示写入日志
    for i, entry in enumerate(log):
        logger.info(f"日志 {i+1}: {entry}")
    
    memory.close()
    return len(log)

if __name__ == "__main__":
    test_write_frequency() 