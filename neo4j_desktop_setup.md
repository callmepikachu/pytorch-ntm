# Neo4j Desktop 设置指南

本指南将帮助您配置Neo4j Desktop以与PyTorch NTM项目配合使用。

## 1. 启动Neo4j Desktop

1. 打开Neo4j Desktop应用程序
2. 创建一个新的数据库项目或使用现有项目
3. 启动一个数据库实例

## 2. 创建数据库

### 方法一：使用Neo4j Desktop界面

1. 在Neo4j Desktop中点击"Add Database"
2. 选择"Create a Local Graph"
3. 设置数据库名称（例如：`ntm-project`）
4. 选择Neo4j版本（推荐5.15或更高）
5. 设置密码（记住这个密码，稍后会用到）
6. 点击"Create"

### 方法二：使用命令行

如果您更喜欢命令行操作：

```bash
# 进入Neo4j Desktop的安装目录
cd ~/Library/Application\ Support/Neo4j\ Desktop/Application/neo4j-desktop-*

# 启动Neo4j服务
./bin/neo4j start
```

## 3. 获取连接信息

### 查看连接详情

在Neo4j Desktop中：
1. 点击您的数据库实例
2. 查看"Connection Details"部分
3. 记录以下信息：
   - **Bolt URI**: 通常是 `bolt://localhost:7687`
   - **HTTP URI**: 通常是 `http://localhost:7474`
   - **用户名**: 默认是 `neo4j`
   - **密码**: 您设置的密码

### 测试连接

在Neo4j Desktop中点击"Start"启动数据库，然后：

1. 点击"Open with Neo4j Browser"
2. 使用您的凭据登录
3. 运行测试查询：
```cypher
RETURN "Hello NTM!" as message
```

## 4. 配置项目环境变量

设置环境变量以连接到您的Neo4j Desktop数据库：

```bash
# 替换为您的实际配置
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password_here"
```

### 永久设置环境变量

#### macOS/Linux:
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export NEO4J_URI="bolt://localhost:7687"' >> ~/.zshrc
echo 'export NEO4J_USER="neo4j"' >> ~/.zshrc
echo 'export NEO4J_PASSWORD="your_password_here"' >> ~/.zshrc
source ~/.zshrc
```

#### Windows:
```cmd
setx NEO4J_URI "bolt://localhost:7687"
setx NEO4J_USER "neo4j"
setx NEO4J_PASSWORD "your_password_here"
```

## 5. 测试连接

运行项目的测试脚本：

```bash
python test_neo4j_setup.py
```

如果测试通过，您会看到：
```
✅ Neo4j 连接成功!
✅ NTM Neo4j 集成测试成功!
🎉 所有测试通过! Neo4j 设置正确。
```

## 6. 开始训练

现在您可以开始使用Neo4j作为长期记忆后端进行训练：

```bash
# 复制任务
python train.py --task copy --model-type dual --long-term-memory-backend neo4j

# CDCL-NLI任务
python train.py --task cdcl-nli --model-type dual --long-term-memory-backend neo4j
```

## 7. 查看训练结果

### 在Neo4j Browser中查看：

1. 在Neo4j Desktop中点击"Open with Neo4j Browser"
2. 运行以下查询查看训练结果：

```cypher
// 查看所有命题节点
MATCH (n:Proposition) RETURN n LIMIT 10

// 查看推理关系
MATCH (n:Proposition)-[:ENTAILS]->(m:Proposition) 
RETURN n, m LIMIT 10

// 查看图结构
MATCH (n:Proposition)-[r:ENTAILS]->(m:Proposition) 
RETURN n, r, m
```

## 8. 故障排除

### 常见问题：

1. **连接被拒绝**
   - 确保Neo4j Desktop中的数据库已启动
   - 检查端口7687是否被占用：`lsof -i :7687`

2. **认证失败**
   - 确认用户名和密码正确
   - 在Neo4j Desktop中重置密码

3. **端口冲突**
   - 如果7687端口被占用，可以在Neo4j Desktop中修改端口
   - 或者停止其他Neo4j实例

4. **内存不足**
   - 在Neo4j Desktop中调整数据库的内存设置
   - 或者减少训练批次大小

### 获取帮助：

- Neo4j Desktop文档：https://neo4j.com/docs/desktop-manual/
- Neo4j Browser查询语言：https://neo4j.com/docs/cypher-manual/

## 9. 性能优化

### 为Neo4j Desktop优化：

1. **创建索引**：
```cypher
CREATE INDEX FOR (n:Proposition) ON (n.text)
```

2. **调整内存设置**：
在Neo4j Desktop的数据库设置中调整：
- 堆内存：1-2GB
- 页面缓存：1GB

3. **定期清理**：
```cypher
// 清理旧数据
MATCH (n:Proposition) WHERE n.created < datetime() - duration({days: 7}) 
DETACH DELETE n
```

## 10. 备份和恢复

### 备份数据库：
在Neo4j Desktop中：
1. 停止数据库
2. 点击"Backup"
3. 选择备份位置

### 恢复数据库：
1. 停止当前数据库
2. 点击"Restore"
3. 选择备份文件

---

现在您已经成功配置了Neo4j Desktop与PyTorch NTM项目的集成！ 