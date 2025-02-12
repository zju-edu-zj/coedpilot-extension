import os
import sys
# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import jsonlines
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from transformers import RobertaTokenizer, RobertaModel
from discriminator.dependency_analyzer import cal_dep_score, DependencyClassifier
from discriminator.siamese_net import evaluate_embedding_model, load_siamese_data
from model_manager import load_model_with_cache

def load_reg_model(lang):
    # 修改模型路径 - 从 analyze.py 位置往上走三级到达项目根目录
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                             'models', f"{lang}/reg_model.pickle")
    with open(model_path, 'rb') as file:
        reg = pickle.load(file)
    print("回归模型的详细信息:")
    print(f"模型类型: {type(reg)}")
    print(f"模型参数: {reg.get_params()}")
    print(f"模型系数: {reg.coef_}")
    print(f"模型截距: {reg.intercept_}")
    print("Successfully loaded discriminator regression model")
    return reg

def load_model(model_path):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = RobertaModel.from_pretrained("huggingface/CodeBERTa-small-v1")
    tokenizer = RobertaTokenizer.from_pretrained(
        "huggingface/CodeBERTa-small-v1")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model, tokenizer, device

MODEL_ROLE = "embedding"

def calculate_similarity(model, tokenizer, target_code, segments, device):
    """一次性计算目标代码与多个代码片段的相似度
    Args:
        model: RoBERTa模型
        tokenizer: RoBERTa分词器
        target_code: 目标代码
        segments: 代码片段列表
        device: 计算设备
    Returns:
        list: 相似度分数列表
    """
    # 编码目标代码
    inputs1 = tokenizer(target_code, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    
    # 编码所有代码片段
    inputs2 = tokenizer(segments, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}
    
    # 计算编码
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    # 获取 [CLS] token的输出
    embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [1, hidden_size]
    embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [n_segments, hidden_size]
    
    # 计算余弦相似度
    similarities = torch.nn.functional.cosine_similarity(embeddings1.repeat(len(segments), 1), embeddings2)
    
    return similarities.tolist()

def calculate_dep_scores(hunk, segments, dependency_analyzer):
    """一次性计算目标代码与多个代码片段的依赖分数
    Args:
        hunk: 目标代码信息
        segments: 代码片段列表
        dependency_analyzer: 依赖分析器
    Returns:
        list: 依赖分数列表
    """
    # 构建所有代码对
    code_pairs = [(hunk['code_window'][0], segment) for segment in segments]
    
    # 使用dependency_analyzer的批处理功能
    corpus_pair = dependency_analyzer.construct_corpus_pair(code_pairs)
    results = dependency_analyzer.batch_gen(corpus_pair)
    
    # 返回最大依赖分数
    return results.tolist()

def analyze_code_differences(input_path, output_path, language='python'):
    """
    分析jsonl文件中每行的代码差异并计算综合评分，将高分片段写入新文件
    Args:
        input_path: jsonl文件的路径
        output_path: 输出文件的路径
        language: 编程语言，默认为python
    """
    # 加载所需模型
    model, tokenizer, device = load_model_with_cache(
        MODEL_ROLE, language, load_model)
    
    # 加载回归模型和依赖分析器
    reg_model = load_reg_model(language)
    dependency_analyzer = DependencyClassifier()
    
    # 添加统计计数器
    total_segments = 0
    filtered_segments_count = 0
    processed_objects = 0
    
    # 读取jsonl文件并写入新文件
    with jsonlines.open(input_path) as reader, \
         jsonlines.open(output_path, mode='w') as writer:
        for idx, obj in enumerate(reader):
            processed_objects += 1
            print(f"\n分析第 {processed_objects} 行数据:")
            code_tokens = obj['code_tokens']
            
            # 提取被mask覆盖的目标代码（保持原始格式）
            target_lines = []
            for line in code_tokens.split('\n'):
                if '<mask>' in line:
                    target_lines.append(line)  # 保持原始格式，包括<mask>和空格
                else:
                    break
            target_code = '\n'.join(target_lines)
            
            # print("\n目标代码片段:")
            # print(target_code)
            
            # 分割其他代码片段并分析，跳过第一个片段（目标代码）
            code_segments = [seg.strip() for seg in code_tokens.split('</s>')[1:] if seg.strip()]
            
            original_count = len(code_segments)
            # 如果代码片段小于4，则跳过
            if original_count <= 4:
                writer.write(obj)
                continue

            total_segments += original_count
            
            # 一次性计算所有片段的依赖分数
            hunk = {'code_window': [target_code, "", ""]}
            dep_scores = calculate_dep_scores(hunk, code_segments, dependency_analyzer)
            
            # 一次性计算所有片段的语义相似度
            similarities = calculate_similarity(model, tokenizer, target_code, code_segments, device)
            
            # 批量计算最终分数
            X = np.array([[d, s] for d, s in zip(dep_scores, similarities)])
            final_scores = reg_model.predict(X)
            
            # 根据分数过滤代码片段
            filtered_segments = []
            for segment, final_score in zip(code_segments, final_scores):
                if final_score >= 0.14:
                    filtered_segments.append(segment)
            
            kept_count = len(filtered_segments)
            filtered_segments_count += kept_count
            
            # 如果有符合条件的片段，构建新的代码tokens并写入
            if filtered_segments:
                # 保持原始格式重建code_tokens
                new_code_tokens = target_code + ' </s> ' + ' </s> '.join(filtered_segments)
                new_obj = obj.copy()
                new_obj['code_tokens'] = new_code_tokens
                writer.write(new_obj)
            
            # 当前行统计信息
            original_count = len(code_segments)
            kept_count = len(filtered_segments)
            print(f"原始代码片段数: {original_count}, 保留片段数: {kept_count}, 过滤掉片段数: {original_count - kept_count}")
            print(final_scores)
            
            # # 输出结果
            # for idx, (segment, dep_score, similarity, final_score) in enumerate(
            #     zip(code_segments, dep_scores, similarities, final_scores)):
            #     print(f"\n代码片段 {idx + 1} 的评分结果:")
            #     print(f"依赖分数: {dep_score:.4f}")
            #     print(f"语义相似度: {similarity:.4f}")
            #     print(f"最终预测分数: {final_score:.4f}")
            #     print(f"是否保留: {'是' if final_score >= 0.5 else '否'}")
            #     print("=" * 50)
    
    # 输出总体统计信息
    print("\n总体统计信息:")
    print(f"处理的数据条数: {processed_objects}")
    print(f"原始代码片段总数: {total_segments}")
    print(f"保留的代码片段数: {filtered_segments_count}")
    print(f"过滤掉的代码片段数: {total_segments - filtered_segments_count}")
    print(f"代码片段保留率: {filtered_segments_count/total_segments*100:.2f}%")
    print(f"\n处理完成，结果已写入: {output_path}")

input_dir = "old_dataset"
output_dir = "new_dataset"

if __name__ == "__main__":
    # 指定要分析的jsonl文件路径
    path = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    for p in path:
        jsonl_path = os.path.join(input_dir, p)
        output_path = os.path.join(output_dir, p)
        # 运行分析
        analyze_code_differences(jsonl_path, output_path)

