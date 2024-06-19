import torch
from transformers import BertModel

class FinetuneBertModel(torch.nn.Module):
    # model_name：指定预训练模型；freeze_bert是否冻结BERT模型参数
    def __init__(self, model_name='D:/mycreate/QA-defense-system-backend/similarity_model/bert-base-chinese', freeze_bert=False):
        super(FinetuneBertModel,self).__init__()
        # bert模型
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad=False
        # 定义一个线性层（全连接层）作为分类器
        # 它接受768维的输入（BERT的隐藏层大小）并输出一个维度为1的向量
        # 用于表示相似度分数。
        self.class_net = torch.nn.Linear(768,1)

    # 微调的具体操作
    def forward(self,input_ids,attention_masks):
        # 将输入传入BERT模型
        outputs = self.bert(input_ids, attention_mask=attention_masks)
        # 获取bert最后一层的隐藏层特征
        last_hidden_state=outputs.last_hidden_state
        # 把token embedding平均得到sentences_embedding
        sentences_embeddings=torch.mean(last_hidden_state,dim=1)
        sentences_embeddings=sentences_embeddings.squeeze(1)
        # 把sentences_embedding输入分类网络
        out=self.class_net(sentences_embeddings).squeeze(-1)
        # 返回模型输出
        return out
