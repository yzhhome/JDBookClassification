import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchsnooper

class Config(object):
    """
    配置参数
    """

    def __init__(self, dataset):
        self.model_name = 'CNN'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.save_path = dataset + '/model/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/logs/' + self.model_name                         # 日志保存路径
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 10000                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 50000                                            # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 32                                            # mini-batch大小
        self.pad_size = 400                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.embed = 300                                                # 向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)        
        self.eps = 1e-8


class Model(nn.Module):
    """
    卷积神经网络用作文本分类
    """

    def __init__(self, config):
        super().__init__()

        # 词向量编码
        self.embedding = nn.Embedding(config.n_vocab, config.embed)

        # 连续使用3个不同kernel_size的卷积
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])

        # dropput层
        self.dropout = nn.Dropout(config.dropout)

        # 使用全连接层把维度降到要分类的数量
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    # 卷积和池化操作
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # 转换词向量
        out = self.embedding(x[0])

        # 增加一个维度送到卷积层        
        out = out.unsqueeze(1)

        # 把卷积层输出进行拼接
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)

        # 再进行dropout
        out = self.dropout(out)

        # 接全连接层输出
        out = self.fc(out)

        return out

