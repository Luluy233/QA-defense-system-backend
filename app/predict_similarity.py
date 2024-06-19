# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BertModel
import torch
from app.FinetuneBertModel import FinetuneBertModel
import os

# 加载预训练模型的分词器
model_name = 'D:/mycreate/QA-defense-system-backend/similarity_model/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型的部分状态字典
checkpoint = torch.load('similarity_epochs8_lr2e-5_batch128/checkpoint')
model_state_dict = checkpoint['model_state_dict']
# 构建一个与预训练模型匹配的 BertModel 实例
bert_model = BertModel.from_pretrained(model_name)

# 从模型状态字典中加载 bert 部分的权重
bert_state_dict = {k.replace('bert.', ''): v for k, v in model_state_dict.items() if k.startswith('bert')}

# 加载 bert 部分的权重到预训练模型
bert_model.load_state_dict(bert_state_dict, strict=False)

# 构建您的 Fine-tune 模型并加载剩余状态字典
model = FinetuneBertModel()
model.bert = bert_model  # 将加载的 bert 部分设置给 FinetuneBertModel 的 bert 属性
model.load_state_dict(model_state_dict, strict=False)  # 加载剩余状态字典
model.eval()

possibility = 0.50

'''
    接口调用模型进行预测
'''


def predict_similarity(text1, text2):

    inputs = tokenizer(text1, text2, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_masks = inputs['attention_mask']

    outputs = model(input_ids=input_ids,
                    attention_masks=attention_masks
                    )
    outputs = torch.sigmoid(outputs).item()

    result = 0 if outputs < possibility else 1
    # return outputs
    return result




'''
    本地测试函数
'''
def main():

    sentence_a = "我想要一杯热咖啡"
    sentence_b = "一杯咖啡，谢谢"
    result = predict_similarity(sentence_a, sentence_b)
    print(result)



if __name__ == '__main__':
    main()