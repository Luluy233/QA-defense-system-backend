# -*- coding: utf-8 -*-
from transformers import BertTokenizer
import torch

from app.FinetuneBertModel import FinetuneBertModel

# 加载预训练模型的分词器
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

possibility = 0.5


'''
    接口调用模型进行预测
'''
def predict_similarity(text1, text2):
    checkpoint = torch.load('app/similarity_bert_lr2e-5/checkpoint')
    model = FinetuneBertModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    inputs = tokenizer(text1, text2, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_masks=inputs['attention_mask']

    outputs = model(input_ids=input_ids,
                   attention_masks=attention_masks
                   )
    outputs = torch.sigmoid(outputs).item()

    result = 0 if outputs > possibility else 1
    # return outputs
    return result;




# '''
#     本地测试函数
# '''
# def main():
#
#     sentence_a = "我想要一杯热咖啡"
#     sentence_b = "一杯咖啡，谢谢"
#     result = predict_similarity(sentence_a, sentence_b)
#     print(result)
#
#
#
# if __name__ == '__main__':
#     main()