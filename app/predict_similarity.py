# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import time

# 加载预训练的BERT模型和tokenizer
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
# 输入文本
# 分词并添加特殊标记

def similarity(text1, text2):
    tokens1 = tokenizer.tokenize(text1)
    tokens1 = ['[CLS]'] + tokens1 + ['[SEP]']
    tokens2 = tokenizer.tokenize(text2)
    tokens2 = ['[CLS]'] + tokens2 + ['[SEP]']

    # 将分词转换为词汇表中的索引
    input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)

    # print(input_ids1)
    # 将输入转换为PyTorch张量
    input_tensor1 = torch.tensor([input_ids1])
    input_tensor2 = torch.tensor([input_ids2])
    # print(input_tensor1)
    # 获取词向量
    with torch.no_grad():
        outputs1 = model(input_tensor1)
        embeddings1 = outputs1[0][0]
        outputs2 = model(input_tensor2)
        embeddings2 = outputs2[0][0]
    # 计算句子表示
    sentence_embedding1 = torch.mean(embeddings1, dim=0)
    sentence_embedding2 = torch.mean(embeddings2, dim=0)
    # print(sentence_embedding1)
    # 计算余弦相似度
    similarity = cosine_similarity(sentence_embedding1.unsqueeze(0), sentence_embedding2.unsqueeze(0))
    if(similarity[0][0] > 0.95):
        return 1
    else:
        return 0


# 相似度预测
def evaluate_model(train_file):
    with open(train_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        text1, text2, label = parts
        label = int(label)
        prediction = similarity(text1, text2)
        #预测正确
        if prediction == label:
            if prediction == 1:
                TP += 1
            else:
                TN += 1
        #预测错误
        else:
            if prediction == 1:
                FP += 1
            else:
                FN += 1

    #准确率
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    #精确率
    precision = TP / (TP + FP)
    #召回率
    recall = TP / (TP + FN)
    # F1-score
    f1_score = 2/(1/precision + 1/recall)

    return accuracy, precision, recall, f1_score


def main():
    # 记录开始时间
    # start_time = time.perf_counter()
    #
    # # 假设train.txt文件位于当前目录
    # train_file = 'dev.txt'
    # accuracy = evaluate_model(train_file)
    # print(f'模型在训练集上的准确度为: {accuracy * 100:.2f}%')
    #
    # # 记录结束时间
    # end_time = time.perf_counter()
    # # 计算运行时间
    # run_time = end_time - start_time
    # print(f"程序运行时间为：{run_time}秒")

    sentence_a = input("请输入句子1：")
    sentence_b = input("请输入句子2：")
    result = similarity(sentence_a, sentence_b)
    print(result)


if __name__ == '__main__':
    main()