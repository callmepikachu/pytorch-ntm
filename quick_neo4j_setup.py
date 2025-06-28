#!/usr/bin/env python
"""
快速Neo4j Desktop配置脚本

使用默认设置快速配置Neo4j Desktop连接。
"""

import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def setup_neo4j_environment():
    """设置Neo4j环境变量"""
    # 默认配置
    config = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "12345678"  # 您需要在Neo4j Desktop中设置这个密码
    }
    
    # 设置环境变量
    os.environ['NEO4J_URI'] = config['uri']
    os.environ['NEO4J_USER'] = config['user']
    os.environ['NEO4J_PASSWORD'] = config['password']
    
    logger.info("🔧 设置环境变量:")
    logger.info(f"  URI: {config['uri']}")
    logger.info(f"  用户: {config['user']}")
    logger.info(f"  密码: {config['password']}")
    
    return config

def save_config_file(config):
    """保存配置文件"""
    config_content = f"""# Neo4j Desktop 配置
# 由 quick_neo4j_setup.py 自动生成

export NEO4J_URI="{config['uri']}"
export NEO4J_USER="{config['user']}"
export NEO4J_PASSWORD="{config['password']}"

# 使用说明:
# 1. 将此文件内容添加到您的 shell 配置文件 (~/.zshrc, ~/.bashrc 等)
# 2. 或者运行: source neo4j_config.sh
"""
    
    with open('neo4j_config.sh', 'w') as f:
        f.write(config_content)
    
    logger.info("💾 配置已保存到 neo4j_config.sh")

def test_neo4j_connection():
    """测试Neo4j连接"""
    try:
        from neo4j import GraphDatabase
        
        logger.info("🧪 测试Neo4j连接...")
        
        driver = GraphDatabase.driver(
            os.environ['NEO4J_URI'], 
            auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
        )
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                logger.info("✅ Neo4j连接测试成功!")
                return True
            else:
                logger.error("❌ Neo4j连接测试失败")
                return False
                
    except ImportError:
        logger.error("❌ 未安装neo4j包，请运行: pip install neo4j")
        return False
    except Exception as e:
        logger.error(f"❌ Neo4j连接失败: {e}")
        logger.info("💡 请检查:")
        logger.info("  1. Neo4j Desktop中的数据库是否已启动")
        logger.info("  2. 密码是否正确（默认: password123）")
        logger.info("  3. 在Neo4j Desktop中重置密码为: password123")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

def test_ntm_integration():
    """测试NTM与Neo4j的集成"""
    try:
        from ntm.long_term_memory import Neo4jGraphMemory
        from ntm.encoder_decoder import encode_text_to_vector, decode_add_vector_to_text
        
        logger.info("🧪 测试NTM Neo4j集成...")
        
        memory = Neo4jGraphMemory(
            uri=os.environ['NEO4J_URI'],
            user=os.environ['NEO4J_USER'],
            password=os.environ['NEO4J_PASSWORD'],
            node_dim=8,
            encoder=encode_text_to_vector,
            decoder=decode_add_vector_to_text,
            log_write=True
        )
        
        # 测试写入和读取
        import torch
        w_t = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]])
        w_prev = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]])
        e = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        a = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        
        memory.write(w_t, w_prev, e, a)
        logger.info("✅ NTM写入测试成功!")
        
        key = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        r_normal, r_forward, r_backward = memory.read(key)
        logger.info("✅ NTM读取测试成功!")
        
        memory.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ NTM集成测试失败: {e}")
        return False

def show_next_steps():
    """显示后续步骤"""
    logger.info("\n🎉 Neo4j Desktop 配置完成!")
    logger.info("\n📋 后续步骤:")
    logger.info("1. 确保Neo4j Desktop中的数据库已启动")
    logger.info("2. 如果密码不是'password123'，请在Neo4j Desktop中重置密码")
    logger.info("3. 运行训练命令:")
    logger.info("   python train.py --task copy --model-type dual --long-term-memory-backend neo4j")
    logger.info("4. 在Neo4j Desktop中点击'Open with Neo4j Browser'查看结果")
    logger.info("5. 运行查询: MATCH (n:Proposition) RETURN n LIMIT 10")
    
    logger.info("\n💡 提示:")
    logger.info("- 如果重启终端，请运行: source neo4j_config.sh")
    logger.info("- 或者将neo4j_config.sh的内容添加到您的shell配置文件")

def main():
    """主函数"""
    logger.info("🚀 快速 Neo4j Desktop 配置")
    logger.info("=" * 40)
    
    # 设置环境变量
    config = setup_neo4j_environment()
    
    # 保存配置文件
    save_config_file(config)
    
    # 测试连接
    if not test_neo4j_connection():
        logger.error("连接测试失败，请检查Neo4j Desktop设置")
        logger.info("💡 在Neo4j Desktop中:")
        logger.info("  1. 确保数据库已启动")
        logger.info("  2. 将密码设置为: password123")
        return
    
    # 测试NTM集成
    if not test_ntm_integration():
        logger.error("NTM集成测试失败")
        return
    
    # 显示后续步骤
    show_next_steps()

if __name__ == "__main__":
    main() 