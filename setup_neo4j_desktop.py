#!/usr/bin/env python
"""
Neo4j Desktop 配置脚本

帮助用户快速配置Neo4j Desktop与PyTorch NTM项目的集成。
"""

import os
import sys
import getpass
import subprocess
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_neo4j_desktop_running():
    """检查Neo4j Desktop是否在运行"""
    try:
        # 检查端口7687是否被占用
        result = subprocess.run(['lsof', '-i', ':7687'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ 检测到Neo4j服务在端口7687运行")
            return True
        else:
            logger.warning("⚠️  未检测到Neo4j服务在端口7687运行")
            return False
    except FileNotFoundError:
        logger.warning("⚠️  无法检查端口状态（lsof命令不可用）")
        return False

def get_neo4j_credentials():
    """获取Neo4j凭据"""
    logger.info("🔐 配置Neo4j连接凭据...")
    
    # 获取连接信息
    uri = input("请输入Neo4j Bolt URI (默认: bolt://localhost:7687): ").strip()
    if not uri:
        uri = "bolt://localhost:7687"
    
    user = input("请输入用户名 (默认: neo4j): ").strip()
    if not user:
        user = "neo4j"
    
    password = getpass.getpass("请输入密码: ")
    if not password:
        logger.error("❌ 密码不能为空")
        return None
    
    return {
        "uri": uri,
        "user": user,
        "password": password
    }

def test_neo4j_connection(credentials):
    """测试Neo4j连接"""
    try:
        from neo4j import GraphDatabase
        
        logger.info("🧪 测试Neo4j连接...")
        
        driver = GraphDatabase.driver(
            credentials["uri"], 
            auth=(credentials["user"], credentials["password"])
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
        return False
    finally:
        if 'driver' in locals():
            driver.close()

def save_environment_variables(credentials):
    """保存环境变量到配置文件"""
    config_content = f"""# Neo4j Desktop 配置
# 由 setup_neo4j_desktop.py 自动生成

export NEO4J_URI="{credentials['uri']}"
export NEO4J_USER="{credentials['user']}"
export NEO4J_PASSWORD="{credentials['password']}"

# 使用说明:
# 1. 将此文件内容添加到您的 shell 配置文件 (~/.zshrc, ~/.bashrc 等)
# 2. 或者运行: source neo4j_config.sh
"""
    
    # 保存到文件
    with open('neo4j_config.sh', 'w') as f:
        f.write(config_content)
    
    logger.info("💾 配置已保存到 neo4j_config.sh")
    
    # 设置当前会话的环境变量
    os.environ['NEO4J_URI'] = credentials['uri']
    os.environ['NEO4J_USER'] = credentials['user']
    os.environ['NEO4J_PASSWORD'] = credentials['password']
    
    logger.info("✅ 环境变量已设置")

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
    logger.info("2. 运行训练命令:")
    logger.info("   python train.py --task copy --model-type dual --long-term-memory-backend neo4j")
    logger.info("3. 在Neo4j Desktop中点击'Open with Neo4j Browser'查看结果")
    logger.info("4. 运行查询: MATCH (n:Proposition) RETURN n LIMIT 10")
    
    logger.info("\n💡 提示:")
    logger.info("- 如果重启终端，请运行: source neo4j_config.sh")
    logger.info("- 或者将neo4j_config.sh的内容添加到您的shell配置文件")

def main():
    """主函数"""
    logger.info("🚀 Neo4j Desktop 配置向导")
    logger.info("=" * 50)
    
    # 检查Neo4j是否运行
    if not check_neo4j_desktop_running():
        logger.warning("请确保Neo4j Desktop中的数据库已启动")
        response = input("是否继续配置? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("配置已取消")
            return
    
    # 获取凭据
    credentials = get_neo4j_credentials()
    if not credentials:
        return
    
    # 测试连接
    if not test_neo4j_connection(credentials):
        logger.error("连接测试失败，请检查Neo4j Desktop设置")
        return
    
    # 保存配置
    save_environment_variables(credentials)
    
    # 测试NTM集成
    if not test_ntm_integration():
        logger.error("NTM集成测试失败")
        return
    
    # 显示后续步骤
    show_next_steps()

if __name__ == "__main__":
    main() 