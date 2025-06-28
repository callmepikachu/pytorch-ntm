#!/usr/bin/env python
"""
Neo4j 设置测试脚本

用于验证Neo4j数据库连接和配置是否正确。
"""

import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    """测试Neo4j连接"""
    try:
        from neo4j import GraphDatabase
        
        # 从环境变量获取配置
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password123")
        
        logger.info(f"尝试连接到 Neo4j: {uri}")
        logger.info(f"用户名: {user}")
        
        # 创建驱动
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # 测试连接
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                logger.info("✅ Neo4j 连接成功!")
                return True
            else:
                logger.error("❌ Neo4j 连接测试失败")
                return False
                
    except ImportError:
        logger.error("❌ 未安装 neo4j 包，请运行: pip install neo4j")
        return False
    except Exception as e:
        logger.error(f"❌ Neo4j 连接失败: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

def test_ntm_neo4j_integration():
    """测试NTM与Neo4j的集成"""
    try:
        from ntm.long_term_memory import Neo4jGraphMemory
        from ntm.encoder_decoder import encode_text_to_vector, decode_add_vector_to_text
        
        logger.info("测试 NTM Neo4j 集成...")
        
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
        
        # 测试写入操作
        import torch
        w_t = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]])
        w_prev = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]])
        e = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        a = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        
        memory.write(w_t, w_prev, e, a)
        logger.info("✅ NTM Neo4j 写入测试成功!")
        
        # 测试读取操作
        key = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        r_normal, r_forward, r_backward = memory.read(key)
        logger.info("✅ NTM Neo4j 读取测试成功!")
        
        # 获取写入日志
        log = memory.get_write_log()
        logger.info(f"写入日志: {log}")
        
        # 导出图数据
        graph_data = memory.export_entailment_graph_json()
        logger.info(f"图数据节点数: {len(graph_data.get('nodes', []))}")
        logger.info(f"图数据边数: {len(graph_data.get('edges', []))}")
        
        memory.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ NTM Neo4j 集成测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始 Neo4j 设置测试...")
    
    # 测试1: Neo4j连接
    if not test_neo4j_connection():
        logger.error("Neo4j 连接测试失败，请检查数据库设置")
        sys.exit(1)
    
    # 测试2: NTM集成
    if not test_ntm_neo4j_integration():
        logger.error("NTM Neo4j 集成测试失败")
        sys.exit(1)
    
    logger.info("🎉 所有测试通过! Neo4j 设置正确。")
    logger.info("\n现在可以运行训练命令:")
    logger.info("python train.py --task copy --model-type dual --long-term-memory-backend neo4j")
    logger.info("python train.py --task cdcl-nli --model-type dual --long-term-memory-backend neo4j")

if __name__ == "__main__":
    main() 