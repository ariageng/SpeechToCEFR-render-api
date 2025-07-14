"""
在其他脚本里这样调用：
from speaking_cefr_predictor_lr import get_speaking_top2_CEFR

# 调用函数
top2_labels = get_speaking_top2_CEFR(85.5, 78.2, 90.1, 82.3, 15, 20)
print(top2_labels)  # 输出: ['B2', 'B1'] (示例，相邻等级)
"""

import joblib
import numpy as np

def get_speaking_top2_CEFR(accuracy_score, completeness_score, confidence_score,
                           fluency_score, new_content, new_delivery):
    """
    预测口语评估的Top-2 CEFR等级（相邻约束）
         
    参数:
        accuracy_score: 准确性分数
        completeness_score: 完整性分数          
        confidence_score: 信心分数
        fluency_score: 流利度分数
        new_content: 新内容分数
        new_delivery: 新表达分数
         
    返回:
        list: Top-2 CEFR标签列表，确保相邻等级
    """
    try:
        # 加载模型和预处理器
        model = joblib.load('lr_model_adjacent.pkl')
        encoder = joblib.load('label_encoder_lr_adjacent.pkl')
        scaler = joblib.load('scaler_lr_adjacent.pkl')
        
        # CEFR等级顺序
        cefr_order = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        
        # 计算total
        avg_score = (accuracy_score + completeness_score + confidence_score + fluency_score) / 4
        total = avg_score + new_content + new_delivery
        
        # 准备并标准化输入数据
        features = np.array([[accuracy_score, completeness_score, confidence_score,
                            fluency_score, new_content, new_delivery, total]])
        features_scaled = scaler.transform(features)
        
        # 预测概率
        probs = model.predict_proba(features_scaled)[0]
        
        # 获取Top-1
        top1_idx = np.argmax(probs)
        top1_label = encoder.inverse_transform([top1_idx])[0]
        
        # 找相邻等级
        try:
            top1_position = cefr_order.index(top1_label)
        except ValueError:
            # 回退到原始top2
            top2_indices = np.argsort(probs)[-2:][::-1]
            return encoder.inverse_transform(top2_indices).tolist()
        
        # 相邻候选
        adjacent_candidates = []
        if top1_position > 0:
            adjacent_candidates.append(cefr_order[top1_position - 1])
        if top1_position < len(cefr_order) - 1:
            adjacent_candidates.append(cefr_order[top1_position + 1])
        
        # 选择概率最高的相邻等级
        best_adjacent_prob = -1
        top2_label = None
        
        for candidate in adjacent_candidates:
            try:
                candidate_idx = encoder.transform([candidate])[0]
                candidate_prob = probs[candidate_idx]
                if candidate_prob > best_adjacent_prob:
                    best_adjacent_prob = candidate_prob
                    top2_label = candidate
            except ValueError:
                continue
        
        # 如果没找到相邻等级，回退
        if top2_label is None:
            top2_indices = np.argsort(probs)[-2:][::-1]
            return encoder.inverse_transform(top2_indices).tolist()
        
        return [top1_label, top2_label]
        
    except FileNotFoundError:
        raise FileNotFoundError("模型文件未找到，请确保逻辑回归模型文件在当前目录")
    except Exception as e:
        raise Exception(f"预测出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 测试函数
    result = get_speaking_top2_CEFR(85.5, 78.2, 90.1, 82.3, 15, 20)
    print(f"Top-2 CEFR预测: {result}")
         
    result2 = get_speaking_top2_CEFR(65.0, 70.5, 75.8, 68.9, 10, 15)
    print(f"Top-2 CEFR预测: {result2}")