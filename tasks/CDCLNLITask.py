"""
CDCL-NLI Task for Dual-Memory NTM.

CDCL-NLI (Contradiction Detection and Correction for Logical Inference)
是一个逻辑推理数据集，需要模型学习逻辑推理链。
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from attr import attrs, attrib, Factory
from transformers import AutoTokenizer, AutoModel

from ntm.aio import EncapsulatedDualMemoryNTM
from ntm.encoder_decoder import encode_text_to_vector, decode_add_vector_to_text


class CDCLNLIDataset(Dataset):
    """CDCL-NLI 数据集加载器 - 支持真实 CDCL-NLI 格式"""
    
    def __init__(self, data_path, tokenizer_name='bert-base-uncased', max_length=128, use_english_only=True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.bert_model = AutoModel.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.use_english_only = use_english_only
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 标签映射
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 处理真实 CDCL-NLI 格式
        # 使用 news1_translated 和 news2_translated 作为前提
        news1 = item.get('news1_translated', '')
        news2 = item.get('news2_translated', '')
        hypothesis = item.get('hypothesis', '')
        label = item.get('label', 'neutral')
        
        # 映射标签
        if label not in self.label_map:
            label = 'neutral'  # 默认值
        label_id = self.label_map[label]
        
        # 合并前提文本
        premise_text = f"{news1} [SEP] {news2}".strip()
        if not premise_text:
            premise_text = "No premise available"
        
        # 合并文本用于序列处理
        combined_text = f"{premise_text} [SEP] {hypothesis}"
        
        # Tokenize
        tokens = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取 BERT embedding
        with torch.no_grad():
            bert_output = self.bert_model(**tokens)
            embeddings = bert_output.last_hidden_state.squeeze(0)  # [seq_len, 768]
        
        return {
            'embeddings': embeddings,
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(label_id, dtype=torch.long)
        }


def cdcl_nli_dataloader(data_path, batch_size=8, max_length=128, shuffle=True, use_english_only=True):
    """CDCL-NLI 数据加载器"""
    dataset = CDCLNLIDataset(data_path, max_length=max_length, use_english_only=use_english_only)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@attrs
class CDCLNLITaskParams:
    """CDCL-NLI 任务参数"""
    name = attrib(default="cdcl-nli-task")
    
    # 模型参数
    controller_size = attrib(default=256, converter=int)
    controller_layers = attrib(default=2, converter=int)
    num_heads = attrib(default=4, converter=int)
    
    # 记忆参数
    short_term_n = attrib(default=256, converter=int)
    short_term_m = attrib(default=64, converter=int)
    long_term_nodes = attrib(default=128, converter=int)
    long_term_dim = attrib(default=128, converter=int)
    
    # 训练参数
    batch_size = attrib(default=8, converter=int)
    num_batches = attrib(default=1000, converter=int)
    num_epochs = attrib(default=10, converter=int)
    learning_rate = attrib(default=1e-4, converter=float)
    max_length = attrib(default=128, converter=int)
    
    # 数据路径
    train_path = attrib(default="data/cdcl-nli/train.json")
    dev_path = attrib(default="data/cdcl-nli/dev.json")
    test_path = attrib(default="data/cdcl-nli/test.json")


@attrs
class CDCLNLITaskModelTraining:
    """CDCL-NLI 任务训练模型"""
    params = attrib(default=Factory(CDCLNLITaskParams))
    projection_layer = attrib()
    net = attrib()
    train_dataloader = attrib()
    dev_dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()
    scheduler = attrib()
    long_term_memory_backend = attrib(default="in-memory")
    neo4j_config = attrib(default=None)
    
    @projection_layer.default
    def default_projection_layer(self):
        """BERT embedding 到模型输入的投影层"""
        return nn.Linear(768, self.params.short_term_m)  # BERT base 是 768 维
    
    @net.default
    def default_net(self):
        """创建双记忆 NTM 模型"""
        # 从环境变量获取Neo4j配置（如果未提供）
        backend = getattr(self, 'long_term_memory_backend', 'in-memory')
        if backend == "neo4j" and getattr(self, 'neo4j_config', None) is None:
            import os
            neo4j_config = {
                "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "user": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password123")
            }
        else:
            neo4j_config = getattr(self, 'neo4j_config', None)
        
        return EncapsulatedDualMemoryNTM(
            input_size=self.params.short_term_m,  # 使用短期记忆维度作为输入
            output_size=3,  # 3个类别：entailment/contradiction/neutral
            controller_size=self.params.controller_size,
            controller_layers=self.params.controller_layers,
            num_heads=self.params.num_heads,
            short_term_memory=(self.params.short_term_n, self.params.short_term_m),
            long_term_memory=(self.params.long_term_nodes, self.params.long_term_dim),
            long_term_memory_backend=backend,
            neo4j_config=neo4j_config,
            encoder=encode_text_to_vector,
            decoder=decode_add_vector_to_text
        )
    
    @train_dataloader.default
    def default_train_dataloader(self):
        return cdcl_nli_dataloader(
            self.params.train_path,
            batch_size=self.params.batch_size,
            max_length=self.params.max_length
        )
    
    @dev_dataloader.default
    def default_dev_dataloader(self):
        return cdcl_nli_dataloader(
            self.params.dev_path,
            batch_size=self.params.batch_size,
            max_length=self.params.max_length,
            shuffle=False
        )
    
    @criterion.default
    def default_criterion(self):
        return nn.CrossEntropyLoss()
    
    @optimizer.default
    def default_optimizer(self):
        # 包含网络和投影层的所有参数
        all_params = list(self.net.parameters()) + list(self.projection_layer.parameters())
        return optim.AdamW(
            all_params,
            lr=self.params.learning_rate,
            weight_decay=0.01
        )
    
    @scheduler.default
    def default_scheduler(self):
        """学习率调度器"""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.params.num_epochs,
            eta_min=1e-6
        )


def train_cdcl_nli_batch(model, batch):
    """训练单个 CDCL-NLI batch"""
    model.optimizer.zero_grad()
    
    embeddings = batch['embeddings']  # [batch_size, seq_len, embed_dim]
    attention_mask = batch['attention_mask']
    labels = batch['label']  # [batch_size]
    
    batch_size = embeddings.size(0)
    seq_len = embeddings.size(1)
    
    # 初始化序列
    model.net.init_sequence(batch_size)
    
    # 使用模型的投影层将 BERT embedding 投影到模型输入维度
    input_vectors = model.projection_layer(embeddings)  # [batch_size, seq_len, short_term_m]
    
    # 调整维度顺序：[batch_size, seq_len, dim] -> [seq_len, batch_size, dim]
    input_vectors = input_vectors.transpose(0, 1)
    
    # 前向传播
    outputs = model.net(input_vectors)  # [seq_len, batch_size, output_size]
    
    # 取最后一个时间步的输出进行分类
    logits = outputs[-1]  # [batch_size, output_size]
    
    # 计算损失
    loss = model.criterion(logits, labels)
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(model.projection_layer.parameters(), max_norm=1.0)
    
    model.optimizer.step()
    
    # 计算准确率
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).float().mean()
    
    return loss.item(), accuracy.item()


def evaluate_cdcl_nli(model, dataloader):
    """评估 CDCL-NLI 模型"""
    model.net.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embeddings']
            labels = batch['label']
            
            batch_size = embeddings.size(0)
            seq_len = embeddings.size(1)
            
            model.net.init_sequence(batch_size)
            
            # 使用模型的投影层
            input_vectors = model.projection_layer(embeddings)
            input_vectors = input_vectors.transpose(0, 1)
            
            outputs = model.net(input_vectors)
            logits = outputs[-1]
            
            loss = model.criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
    
    model.net.train()
    return total_loss / num_batches, total_accuracy / num_batches 