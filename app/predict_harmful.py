
# coding=utf-8

import logging
import torch
import numpy as np
from transformers import BertTokenizer
import argparse

from . import mymodel
# import utils

logger = logging.getLogger(__name__)

def load_model(model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        return model

def predict(tokenizer, text, id2label, args, model,device):
        model.eval()
        model.to(device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=args.max_seq_len,
                                           truncation='longest_first',
                                           padding="max_length",
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
            token_ids = inputs['input_ids'].to(device)
            attention_masks = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            outputs = model(token_ids, attention_masks, token_type_ids)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten().tolist()
            print("output:",outputs)
            if len(outputs) != 0:
                outputs = [id2label[i] for i in outputs]
                return outputs
            else:
                return '不好意思，我没有识别出来'



def predict_harm(model_name,text):
    """
        预测句子是否有害
        params:
            model_name (string): 模型名称
            text (string): 待预测句子

        return:
            boolean: true(negative)
                     false(positive)

        example:
            >>> predict_harm('best,pt',"空姐你好，可以给我拿杯水吗？")
            true
    """
    args = argparse.Namespace(adam_epsilon=1e-08,
                              # bert_dir='../model_hub/chinese-bert-wwm-ext/',
                              #flask运行的当前目录不一样！！！
                              bert_dir='model_hub/chinese-bert-wwm-ext/',
                              data_dir='./data/offensive/',
                              data_name='train',
                              do_predict=False,
                              do_test=False,
                              do_train=False,
                              dropout_prob=0.1,
                              eval_batch_size=16,
                              gpu_ids='0',
                              log_dir='logs/',
                              lr=3e-05,
                              max_grad_norm=1,
                              max_seq_len=256,
                              num_tags=2,
                              other_lr=0.0003,
                              output_dir='./checkpoints/', retrain=False, seed=123, swa_start=3, train_batch_size=16, train_epochs=5, warmup_proportion=0.1, weight_decay=0.01)

    # # 设置随机数种子
    # utils.set_seed(args.seed)
    # # 设置日志记录
    # utils.set_logger(os.path.join(args.log_dir, 'main.log'))

    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mymodel.BertForSequenceClassification(args)

    # checkpoint_path = './checkpoints/train/{}'.format(model_name)
    checkpoint_path = 'app/checkpoints/train/{}'.format(model_name)

    model = load_model(model, checkpoint_path)

    id2label = {0: "positive", 1: "negative"}

    result=predict(tokenizer=tokenizer,text=text,id2label=id2label,args=args,model=model,device=device)

    print("输入数据的判断结果为：",result[0])

    return result[0]


if __name__ == '__main__':
    result = predict_harm('best.pt',"空姐你好，可以给我一杯橙汁吗？")
    print(result)




