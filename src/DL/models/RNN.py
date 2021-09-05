import torch
import torch.nn as nn
import numpy as np


class Config(object):
    """
    配置参数
    """

    def __init__(self, dataset):
        self.model_name = 'RNN'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.save_path = dataset + '/model/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/logs/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 10000                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 50000                                            # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 32                                            # mini-batch大小
        self.pad_size = 400                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.embed = 300                                                # 词向量维度
        self.hidden_size = 512                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.eps = 1e-8

class Model(nn.Module):
    """
    循环神经网络用作文本分类
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out