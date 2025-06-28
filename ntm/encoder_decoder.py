import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import numpy as np

class SimpleTextVectorEncoder:
    """
    使用 transformers（如 SimCSE/BERT）将文本编码为向量。
    """
    def __init__(self, model_name='princeton-nlp/sup-simcse-bert-base-uncased', device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs, output_hidden_states=True, return_dict=True)
            # 取 [CLS] 向量
            vec = output.last_hidden_state[:, 0, :].squeeze(0)
        return vec.cpu().numpy()

class SimpleVectorTextDecoder:
    """
    使用 transformers（如 T5/BART）将向量解码为文本。
    """
    def __init__(self, model_name='t5-small', device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device

    def __call__(self, vector):
        # 简单实现：将向量映射为 token embedding，再解码
        # 这里只做演示，实际可用更复杂的 mapping
        vector = torch.tensor(vector, dtype=torch.float).unsqueeze(0).to(self.device)
        # 生成伪输入（如 "decode:"），并将向量拼接
        input_ids = self.tokenizer("decode:", return_tensors='pt').input_ids.to(self.device)
        # 直接用 decoder_start_token_id 生成
        output_ids = self.model.generate(input_ids, max_length=16)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text

# 工具函数

def encode_text_to_vector(text, encoder=None):
    if encoder is None:
        encoder = SimpleTextVectorEncoder()
    return encoder(text)

def decode_add_vector_to_text(vector, decoder=None):
    if decoder is None:
        # 改进的简单解码器：根据向量值生成描述性文本
        if isinstance(vector, (list, np.ndarray)):
            vector = np.array(vector)
        
        # 根据向量的统计特征生成文本
        mean_val = np.mean(vector)
        std_val = np.std(vector)
        max_val = np.max(vector)
        min_val = np.min(vector)
        
        # 生成描述性文本
        if mean_val > 0.5:
            sentiment = "positive"
        elif mean_val < 0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        if std_val > 0.2:
            complexity = "complex"
        else:
            complexity = "simple"
        
        text = f"vector({sentiment},{complexity},mean={mean_val:.2f})"
        return text
    else:
        return decoder(vector) 