# PyTorch Neural Turing Machine (NTM)

PyTorch implementation of [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (NTM).

An **NTM** is a memory augumented neural network (attached to external memory) where the interactions with the external memory (address, read, write) are done using differentiable transformations. Overall, the network is end-to-end differentiable and thus trainable by a gradient based optimizer.

The NTM is processing input in sequences, much like an LSTM, but with additional benfits: (1) The external memory allows the network to learn algorithmic tasks easier (2) Having larger capacity, without increasing the network's trainable parameters.

The external memory allows the NTM to learn algorithmic tasks, that are much harder for LSTM to learn, and to maintain an internal state much longer than traditional LSTMs.

## 📋 项目改进记录

### 2025年6月28 - 重大架构升级

#### 🎯 核心功能改进
- **双重记忆架构**: 实现了短期记忆（NTM Memory）+ 长期记忆（图记忆）的双重记忆架构
- **长期记忆抽象接口**: 创建了`AbstractGraphMemory`接口，支持多种后端实现
- **Neo4j图数据库集成**: 实现了`Neo4jGraphMemory`，支持将训练过程持久化到图数据库
- **内存后端**: 保留了`InMemoryGraphMemory`作为轻量级选项

#### 🔧 技术架构升级
- **模块化设计**: 将长期记忆抽象为接口层，支持插件式后端切换
- **文本向量转换**: 实现了基于transformers的命题向量与文本相互转换模块
- **双重控制器**: 开发了`DualMemoryController`，支持短期和长期记忆的协同工作
- **编码解码器**: 实现了智能的向量解码器，生成有意义的文本描述

#### 📊 新增任务支持
- **CDCL-NLI任务**: 完整实现了自然语言推理任务，支持官方真实数据格式
- **BERT集成**: 使用BERT embedding作为输入，添加投影层动态匹配模型维度
- **多任务训练**: 支持copy、repeat-copy、cdcl-nli等多种任务

#### 🛠️ 开发工具改进
- **Neo4j Desktop支持**: 提供了完整的Neo4j Desktop配置指南和自动化脚本
- **测试框架**: 创建了完整的测试套件，包括Neo4j集成测试
- **日志系统**: 实现了详细的训练日志记录和图数据导出功能
- **Docker支持**: 提供了docker-compose配置，简化Neo4j部署

#### 🚀 性能优化
- **批量处理**: 支持批量训练和推理
- **内存管理**: 优化了长期记忆的内存使用
- **连接池**: Neo4j连接优化，支持长时间训练
- **错误处理**: 增强了错误处理和恢复机制

#### 📈 训练改进
- **参数覆盖**: 支持通过`-p`参数动态覆盖训练参数
- **检查点系统**: 改进了模型检查点保存和恢复
- **进度监控**: 实时训练进度显示和性能监控
- **多后端支持**: 支持内存和Neo4j后端的无缝切换

## 🚀 快速开始

### 常用运行命令

#### 基础任务训练
```bash
# 传统NTM复制任务
python train.py --task copy --seed 1000

# 双重记忆复制任务（内存后端）
python train.py --task copy --model-type dual --seed 1000

# 双重记忆复制任务（Neo4j后端）
python train.py --task copy --model-type dual --long-term-memory-backend neo4j

# 重复复制任务
python train.py --task repeat-copy --model-type dual --long-term-memory-backend neo4j
```

#### CDCL-NLI任务训练
```bash
# CDCL-NLI任务（内存后端）
python train.py --task cdcl-nli --model-type dual -p num_epochs=5

# CDCL-NLI任务（Neo4j后端）
python train.py --task cdcl-nli --model-type dual --long-term-memory-backend neo4j -p num_epochs=5

# 自定义批次大小和训练轮数
python train.py --task cdcl-nli --model-type dual --long-term-memory-backend neo4j -p num_epochs=10 -p batch_size=16
```

#### 快速测试和验证
```bash
# 测试Neo4j连接和集成
python test_neo4j_setup.py

# 测试写入频率和条件
python test_training_write_conditions.py

# 快速训练测试（少量批次）
python train.py --task copy --model-type dual --long-term-memory-backend neo4j --num-batches 100 --batch-size 2
```

#### 参数调优示例
```bash
# 调整控制器大小和层数
python train.py --task copy --model-type dual --long-term-memory-backend neo4j -p controller_size=200 -p controller_layers=2

# 调整记忆大小
python train.py --task copy --model-type dual --long-term-memory-backend neo4j -p short_term_n=256 -p short_term_m=32

# 调整长期记忆节点数
python train.py --task copy --model-type dual --long-term-memory-backend neo4j -p long_term_nodes=64 -p long_term_dim=16
```

### 使用Neo4j作为长期记忆后端

#### 方法一：Neo4j Desktop（推荐本地用户）

如果您已安装Neo4j Desktop：

1. **启动Neo4j Desktop并创建数据库**：
   - 打开Neo4j Desktop
   - 创建新数据库或启动现有数据库
   - 记住您的密码

2. **运行配置向导**：
```bash
python setup_neo4j_desktop.py
```

3. **开始训练**：
```bash
# 复制任务
python train.py --task copy --model-type dual --long-term-memory-backend neo4j

# CDCL-NLI任务
python train.py --task cdcl-nli --model-type dual --long-term-memory-backend neo4j
```

4. **查看结果**：
   - 在Neo4j Desktop中点击"Open with Neo4j Browser"
   - 运行查询：`MATCH (n:Proposition) RETURN n LIMIT 10`

#### 方法二：Docker（推荐服务器用户）

1. **启动Neo4j数据库**：
```bash
# 使用Docker Compose（推荐）
docker-compose up -d

# 或使用Docker
docker run --name neo4j-ntm -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password123 -d neo4j:5.15
```

2. **设置环境变量**：
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password123"
```

3. **测试设置**：
```bash
python test_neo4j_setup.py
```

4. **开始训练**：
```bash
# 复制任务
python train.py --task copy --model-type dual --long-term-memory-backend neo4j

# CDCL-NLI任务
python train.py --task cdcl-nli --model-type dual --long-term-memory-backend neo4j
```

5. **查看训练结果**：
- 访问 http://localhost:7474 查看Neo4j Browser
- 运行查询：`MATCH (n:Proposition) RETURN n LIMIT 10`

## A PyTorch Implementation

This repository implements a vanilla NTM in a straight forward way. The following architecture is used:

![NTM Architecture](./images/ntm.png)

### Features
* Batch learning support
* Numerically stable
* Flexible head configuration - use X read heads and Y write heads and specify the order of operation
* **copy** and **repeat-copy** experiments agree with the paper
* **Dual Memory Architecture**: 支持短期记忆（NTM Memory）和长期记忆（图记忆）的双重记忆架构
* **Neo4j 图数据库支持**: 可以将训练过程中的中间结果持久化到Neo4j图数据库中
* **CDCL-NLI 任务支持**: 支持自然语言推理任务

***

## Neo4j 图数据库设置

本项目支持使用Neo4j图数据库作为长期记忆的后端存储，可以将训练过程中的命题和推理关系持久化到数据库中。

### 1. 安装Neo4j数据库

#### 方法一：使用Docker Compose（最推荐）
```bash
# 使用Docker Compose启动Neo4j
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs neo4j

# 停止服务
docker-compose down
```

#### 方法二：使用Docker（推荐）
```bash
# 拉取Neo4j官方镜像
docker pull neo4j:5.15

# 启动Neo4j容器
docker run \
    --name neo4j-ntm \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password123 \
    -e NEO4J_PLUGINS='["apoc"]' \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_dbms_security_procedures_unrestricted=apoc.\* \
    --env NEO4J_dbms_memory_heap_initial__size=512m \
    --env NEO4J_dbms_memory_heap_max__size=2G \
    -d neo4j:5.15
```

#### 方法三：本地安装
1. 访问 [Neo4j官网](https://neo4j.com/download/) 下载社区版
2. 解压并安装
3. 启动Neo4j服务：
```bash
# Linux/Mac
./bin/neo4j start

# Windows
bin\neo4j.bat start
```

### 2. 配置Neo4j连接

启动Neo4j后，可以通过以下方式配置连接：

1. **访问Neo4j Browser**: 打开浏览器访问 `http://localhost:7474`
2. **初始登录**: 用户名 `neo4j`，密码 `neo4j`（首次登录会要求修改密码）
3. **修改密码**: 按提示设置新密码（例如：`password123`）

### 3. 验证连接

在Neo4j Browser中运行以下命令验证数据库正常工作：
```cypher
CREATE (n:Test {name: "Hello Neo4j"}) RETURN n
```

#### 使用测试脚本验证设置

项目提供了测试脚本来验证Neo4j设置：
```bash
# 运行Neo4j设置测试
python test_neo4j_setup.py
```

测试脚本会检查：
- Neo4j数据库连接
- NTM与Neo4j的集成
- 读写操作
- 图数据导出功能

如果测试通过，您就可以开始使用Neo4j作为长期记忆后端进行训练了。

### 4. 使用Neo4j作为长期记忆后端

#### 方法一：通过环境变量配置
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password123"
```

#### 方法二：在代码中直接配置
在训练脚本中指定Neo4j配置：
```python
neo4j_config = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j", 
    "password": "password123"
}
```

### 5. 运行训练（使用Neo4j）

```bash
# 使用Neo4j作为长期记忆后端的复制任务
python train.py --task copy --model-type dual --long-term-memory-backend neo4j

# 使用Neo4j的CDCL-NLI任务
python train.py --task cdcl-nli --model-type dual --long-term-memory-backend neo4j
```

### 6. 查看训练结果

训练过程中，模型会将命题和推理关系写入Neo4j数据库。可以通过以下方式查看：

#### 在Neo4j Browser中查询：
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

#### 实用查询示例：
```cypher
// 统计节点和关系数量
MATCH (n) RETURN labels(n) AS labels, count(*) AS count

// 查看所有Proposition节点内容
MATCH (n:Proposition) RETURN n.text AS text, n.embedding AS embedding, n.idx AS idx ORDER BY n.text

// 查看图结构（简化版）
MATCH (n:Proposition)-[r:ENTAILS]->(m:Proposition) 
RETURN n.text AS source, m.text AS target 
ORDER BY n.text, m.text

// 统计关系类型
MATCH ()-[r]->() RETURN type(r) AS relationship_type, count(*) AS count

// 查看训练日志（如果有）
MATCH (n:TrainingLog) RETURN n.source_idx AS source_idx, n.proposition AS proposition, n.timestamp AS timestamp ORDER BY n.timestamp DESC LIMIT 10

// 分析节点连接度
MATCH (n:Proposition)
RETURN n.text AS text, 
       size((n)-[:ENTAILS]->()) AS out_degree,
       size(()-[:ENTAILS]->(n)) AS in_degree
ORDER BY out_degree DESC, in_degree DESC

// 查找最活跃的节点（连接数最多）
MATCH (n:Proposition)
WITH n, size((n)-[:ENTAILS]->()) + size(()-[:ENTAILS]->(n)) AS total_connections
RETURN n.text AS text, total_connections
ORDER BY total_connections DESC LIMIT 10
```

#### 导出图数据：
```python
# 在代码中导出图数据为JSON
memory = model.net.long_term
graph_data = memory.export_entailment_graph_json("entailment_graph.json")
```

### 7. 清理数据

如需清理数据库中的训练数据：
```cypher
// 删除所有命题节点和关系
MATCH (n:Proposition) DETACH DELETE n
```

***

## Copy Task

The **Copy** task tests the NTM's ability to store and recall a long sequence of arbitrary information. The input to the network is a random sequence of bits, ending with a delimiter. The sequence lengths are randomised between 1 to 20.

### Training

Training convergence for the **copy task** using 4 different seeds (see the [notebook](./notebooks/copy-task-plots.ipynb) for details)

![NTM Convergence](./images/copy-train.png)

 The following plot shows the cost per sequence length during training. The network was trained with `seed=10` and shows fast convergence. Other seeds may not perform as well but should converge in less than 30K iterations.

![NTM Convergence](./images/copy-train2.png)

### Evaluation

Here is an animated GIF that shows how the model generalize. The model was evaluated after every 500 training samples, using the target sequence shown in the upper part of the image. The bottom part shows the network output at any given training stage.

![Copy Task](./images/copy-train-20-fast.gif)

The following is the same, but with `sequence length = 80`. Note that the network was trained with sequences of lengths 1 to 20.

![Copy Task](./images/copy-train-80-fast.gif)

***
## Repeat Copy Task

The **Repeat Copy** task tests whether the NTM can learn a simple nested function, and invoke it by learning to execute a __for loop__. The input to the network is a random sequence of bits, followed by a delimiter and a scalar value that represents the number of repetitions to output. The number of repetitions, was normalized to have zero mean and variance of one (as in the paper). Both the length of the sequence and the number of repetitions are randomised between 1 to 10.

### Training

Training convergence for the **repeat-copy task** using 4 different seeds (see the [notebook](./notebooks/repeat-copy-task-plots.ipynb) for details)

![NTM Convergence](./images/repeat-copy-train.png)

### Evaluation

The following image shows the input presented to the network, a sequence of bits + delimiter + num-reps scalar. Specifically the sequence length here is eight and the number of repetitions is five.

![Repeat Copy Task](./images/repeat-copy-ex-inp.png)

And here's the output the network had predicted:

![Repeat Copy Task](./images/repeat-copy-ex-outp.png)

Here's an animated GIF that shows how the network learns to predict the targets. Specifically, the network was evaluated in each checkpoint saved during training with the same input sequence.

![Repeat Copy Task](./images/repeat-copy-train-10.gif)

## Installation

The NTM can be used as a reusable module, currently not packaged though.

1. Clone repository
2. Install [PyTorch](http://pytorch.org/)
3. pip install -r requirements.txt

## Usage

Execute ./train.py

```
usage: train.py [-h] [--seed SEED] [--task {copy,repeat-copy,cdcl-nli}] [-p PARAM]
                [--checkpoint-interval CHECKPOINT_INTERVAL]
                [--checkpoint-path CHECKPOINT_PATH]
                [--report-interval REPORT_INTERVAL]
                [--model-type {ntm,dual}] [--num-batches NUM_BATCHES]
                [--batch-size BATCH_SIZE] [--long-term-memory-backend {in-memory,neo4j}]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Seed value for RNGs
  --task {copy,repeat-copy,cdcl-nli}
                        Choose the task to train (default: copy)
  -p PARAM, --param PARAM
                        Override model params. Example: "-pbatch_size=4
                        -pnum_heads=2 -pnum_epochs=5"
  --checkpoint-interval CHECKPOINT_INTERVAL
                        Checkpoint interval (default: 1000). Use 0 to disable
                        checkpointing
  --checkpoint-path CHECKPOINT_PATH
                        Path for saving checkpoint data (default: './checkpoints')
  --report-interval REPORT_INTERVAL
                        Reporting interval
  --model-type {ntm,dual}
                        Choose model type: ntm or dual-memory
  --num-batches NUM_BATCHES
                        Override number of batches to train
  --batch-size BATCH_SIZE
                        Override batch size
  --long-term-memory-backend {in-memory,neo4j}
                        Choose long-term memory backend: in-memory or neo4j
```

### 使用示例

#### 基础任务训练
```bash
# 传统NTM复制任务
python train.py --task copy --seed 1000

# 双重记忆架构的复制任务
python train.py --task copy --model-type dual --seed 1000

# 使用Neo4j作为长期记忆后端
python train.py --task copy --model-type dual --long-term-memory-backend neo4j

# 重复复制任务
python train.py --task repeat-copy --model-type dual --long-term-memory-backend neo4j
```

#### CDCL-NLI任务训练
```bash
# CDCL-NLI任务（内存后端）
python train.py --task cdcl-nli --model-type dual -p num_epochs=5

# CDCL-NLI任务（Neo4j后端）
python train.py --task cdcl-nli --model-type dual --long-term-memory-backend neo4j -p num_epochs=5

# 自定义批次大小和训练轮数
python train.py --task cdcl-nli --model-type dual --long-term-memory-backend neo4j -p num_epochs=10 -p batch_size=16
```

#### 快速测试和验证
```bash
# 测试Neo4j连接和集成
python test_neo4j_setup.py

# 测试写入频率和条件
python test_training_write_conditions.py

# 快速训练测试（少量批次）
python train.py --task copy --model-type dual --long-term-memory-backend neo4j --num-batches 100 --batch-size 2
```

#### 参数调优示例
```bash
# 调整控制器大小和层数
python train.py --task copy --model-type dual --long-term-memory-backend neo4j -p controller_size=200 -p controller_layers=2

# 调整记忆大小
python train.py --task copy --model-type dual --long-term-memory-backend neo4j -p short_term_n=256 -p short_term_m=32

# 调整长期记忆节点数
python train.py --task copy --model-type dual --long-term-memory-backend neo4j -p long_term_nodes=64 -p long_term_dim=16

# 自定义参数组合
python train.py --task copy --model-type dual -p batch_size=8 -p num_heads=4 -p controller_size=150
```

## 项目结构

```
pytorch-ntm/
├── ntm/                    # 核心NTM实现
│   ├── ntm.py             # 基础NTM和双重记忆NTM
│   ├── memory.py          # 短期记忆实现
│   ├── long_term_memory.py # 长期记忆接口和实现
│   ├── controller.py      # 控制器实现
│   ├── dual_controller.py # 双重记忆控制器
│   ├── head.py           # 读写头实现
│   └── encoder_decoder.py # 文本编码解码器
├── tasks/                 # 任务实现
│   ├── copytask.py       # 复制任务
│   ├── repeatcopytask.py # 重复复制任务
│   ├── DualCopyTask.py   # 双重记忆复制任务
│   └── CDCLNLITask.py    # CDCL-NLI任务
├── data/                  # 数据集
│   └── cdcl-nli/         # CDCL-NLI数据集
├── checkpoints/          # 模型检查点
├── tests/               # 测试文件
└── train.py            # 训练脚本
```

### 获取帮助

- Neo4j Desktop文档：https://neo4j.com/docs/desktop-manual/
- Neo4j Browser查询语言：https://neo4j.com/docs/cypher-manual/
- 项目问题反馈：请提交GitHub Issue

## 故障排除

### Neo4j连接问题

1. **连接被拒绝**: 确保Neo4j服务正在运行
   ```bash
   # 检查Neo4j状态
   docker ps | grep neo4j
   # 或
   neo4j status
   ```

2. **认证失败**: 检查用户名和密码是否正确
   ```bash
   # 重置密码
   docker exec -it neo4j-ntm neo4j-admin set-initial-password newpassword
   ```

3. **端口冲突**: 确保7474和7687端口未被占用
   ```bash
   # 检查端口占用
   lsof -i :7474
   lsof -i :7687
   ```

### Neo4j Desktop特定问题

1. **数据库未启动**: 在Neo4j Desktop中点击"Start"启动数据库
2. **密码错误**: 在Neo4j Desktop中重置数据库密码
3. **端口被占用**: 检查Neo4j Desktop的数据库设置，确认端口配置

### 训练相关问题

1. **参数错误**: 使用`-p`参数覆盖时确保参数名正确
   ```bash
   # 正确用法
   python train.py --task cdcl-nli --model-type dual -p num_epochs=5
   
   # 错误用法（缺少等号）
   python train.py --task cdcl-nli --model-type dual -p num_epochs 5
   ```

2. **内存不足**: 调整批次大小和模型参数
   ```bash
   # 减小批次大小
   python train.py --task cdcl-nli --model-type dual -p batch_size=4
   
   # 减小模型大小
   python train.py --task copy --model-type dual -p controller_size=50 -p short_term_n=64
   ```

3. **Neo4j写入较少**: 这是正常现象，可以：
   - 运行更长的训练：`--num-batches 1000`
   - 使用测试脚本验证：`python test_training_write_conditions.py`
   - 检查Neo4j连接：`python test_neo4j_setup.py`

### 内存不足

如果遇到内存不足问题，可以调整Neo4j的内存设置：
```bash
# 在docker run命令中添加
--env NEO4J_dbms_memory_heap_initial__size=256m \
--env NEO4J_dbms_memory_heap_max__size=1G \
```

### 性能优化

1. **批量操作**: Neo4j支持批量写入，可以提高性能
2. **索引优化**: 为Proposition节点的text属性创建索引
   ```cypher
   CREATE INDEX FOR (n:Proposition) ON (n.text)
   ```
3. **内存配置**: 根据系统内存调整Neo4j的堆内存设置