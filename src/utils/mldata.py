import numpy as np
from numpy.core.fromnumeric import trace
import pandas as pd
import json
import os
from ML.sklearn_models_word2vec import sentence2vec
from src.utils import config
from src.utils.tools import create_logger, sentence_embedding, query_cut
from src.word2vec.embedding import Embedding

logger = create_logger(config.log_dir + 'ml_data.log')

class MLData(object):
    """
    机器学习模型数据处理类
    """
    def __init__(self, debug_model=False, train_mode=True):
        """
        加载embedding， 如果不训练， 则不处理数据
        """
        self.debug_model = debug_model
        self.train_model = train_mode
        self.embedding = Embedding()
        self.embedding.load_model()

        if train_mode:
            self.preprocessor()

    def preprocessor(self):
        logger.info("load data")

        self.train = pd.read_csv(config.root_path + '/data/train_clean.csv', sep='\t').dropna()
        self.dev = pd.read_csv(config.root_path + '/data/test_clean.csv', sep='\t').dropna()

        # 调试模式只取少量数据
        if self.debug_model:
            self.train = self.train.sample(n=1000).reset_index(drop=True)
            self.dev = self.dev.sample(n=100).reset_index(drop=True)

        # 分词
        self.train["queryCut"] = self.train["text"].apply(query_cut)
        self.dev["queryCut"] = self.dev["text"].apply(query_cut)

        # 过滤掉停用词
        self.train['queryCutRMStopWord'] = self.train['queryCut'].apply(
            lambda x: [word for word in x if word not in self.embedding.stopwords])

        self.dev['queryCutRMStopWord'] = self.train['queryCut'].apply(
            lambda x: [word for word in x if word not in self.embedding.stopwords])

        # 存在label2id.json则直接加载
        if os.path.exists(config.root_path + '/data/label2id.json'):
            labelNameToIndex = json.load(
                open(config.root_path + '/data/label2id.json', encoding='utf-8'))
        # 不存在则生成label2id.json
        else:
            # 标签列表
            labelName = self.train['label'].unique()

            # 索引列表
            labelIndex = list(range(labelName))

            # 组成标签和索引字典
            labelNameToIndex = dict(zip(labelName, labelIndex))
            
            with open(config.root_path + 'data/label2id.json', 'w', 
                    encoding='utf-8') as f:
                json.dump({k: v for k, v in labelNameToIndex.items()}, f)

        # label名字映射到标签并保存到列labelIndex中
        self.train['labelIndex'] = self.train['label'].map(labelNameToIndex)
        self.dev['labelIndex'] = self.train['label'].map(labelNameToIndex)
        

    def process_data(self, method='word2vec'):
        """
        生成特征向量化后的训练集和测试集数据
        """
        X_train = self.get_feature(self.train, method)
        X_test = self.get_feature(self.dev, method)

        y_train = self.train["labelIndex"]
        y_test = self.dev["labelIndex"]

        return X_train, X_test, y_train, y_test

    def get_feature(self, data, method='word2vec'):
        """
        获取向量化后的特征
        """
        if method == 'tfidf':
            data = [' '.join(query) for query in data['queryCutRMStopWord']]
            return self.embedding.tfidf_model.transform(data)
        elif method == 'word2vec':
            return np.vstack(data['queryCutRMStopWord'].apply(
                lambda x: sentence_embedding(x, self.embedding.w2v_model)[0]))
        elif method == 'fasttext':
            return np.vstack(data['queryCutRMStopWord'].apply(
                lambda x: sentence_embedding(x, self.embedding.fast_model)[0]))        
        else:
            raise NotImplementedError