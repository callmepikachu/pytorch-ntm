#!/usr/bin/env python
"""
测试改进后的向量解码器
"""

import numpy as np
from ntm.encoder_decoder import decode_add_vector_to_text

def test_decoder():
    """测试解码器"""
    print("🧪 测试改进后的向量解码器")
    print("=" * 40)
    
    # 测试向量
    test_vectors = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 高均值，高方差
        [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8],  # 高均值，低方差
        [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],  # 低均值，低方差
        [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],  # 中等均值，高方差
    ]
    
    for i, vector in enumerate(test_vectors):
        text = decode_add_vector_to_text(vector)
        mean_val = np.mean(vector)
        std_val = np.std(vector)
        print(f"向量 {i+1}: {vector}")
        print(f"  均值: {mean_val:.3f}, 标准差: {std_val:.3f}")
        print(f"  解码文本: {text}")
        print()

if __name__ == "__main__":
    test_decoder() 