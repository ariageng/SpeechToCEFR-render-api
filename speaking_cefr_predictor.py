"""
在其他脚本里这样调用：
from speaking_cefr_predictor import get_speaking_top2_CEFR

# 调用函数
top2_labels = get_speaking_top2_CEFR(85.5, 78.2, 90.1, 82.3, 15, 20, 35)
print(top2_labels)  # 输出: ['B2', 'B1'] (示例)
"""

import joblib
import numpy as np

def get_speaking_top2_CEFR(accuracy_score, completeness_score, confidence_score, 
                          fluency_score, new_content, new_delivery):
    """
    预测口语评估的Top-2 CEFR等级
    
    参数:
        accuracy_score: 准确性分数
        completeness_score: 完整性分数  
        confidence_score: 信心分数
        fluency_score: 流利度分数
        new_content: 新内容分数
        new_delivery: 新表达分数
    
    返回:
        list: Top-2 CEFR标签列表，按概率从高到低排序
    """
    try:
        # 加载模型
        model = joblib.load('rf_model_top2.pkl')
        encoder = joblib.load('label_encoder.pkl')
        
        # 计算total: 四项平均分 + new_content + new_delivery
        avg_score = (accuracy_score + completeness_score + confidence_score + fluency_score) / 4
        total = avg_score + new_content + new_delivery
        
        # 准备输入数据
        features = np.array([[accuracy_score, completeness_score, confidence_score,
                            fluency_score, new_content, new_delivery, total]])
        
        # 预测概率
        probs = model.predict_proba(features)
        
        # 获取Top-2标签索引（按概率降序）
        top2_indices = np.argsort(probs[0])[-2:][::-1]
        
        # 转换为原始标签
        top2_labels = encoder.inverse_transform(top2_indices).tolist()
        
        return top2_labels
        
    except FileNotFoundError:
        raise FileNotFoundError("模型文件未找到，请确保 rf_model_top2.pkl 和 label_encoder.pkl 在当前目录")
    except Exception as e:
        raise Exception(f"预测出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 测试函数
    result = get_speaking_top2_CEFR(85.5, 78.2, 90.1, 82.3, 15, 20)
    print(f"Top-2 CEFR预测: {result}")
    
    result2 = get_speaking_top2_CEFR(65.0, 70.5, 75.8, 68.9, 10, 15)
    print(f"Top-2 CEFR预测: {result2}")
