"""
词向量训练功能模块
可以训练tf-idf、word2vec、fasttext、lda四种类型的词向量
"""

import numpy as np
import pandas as pd
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from src.utils import config
from src.utils.config import root_path
from src.utils.tools import create_logger
import gensim
from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
from src.word2vec.autoencoder import AutoEncoder

logger = create_logger(root_path + '/logs/embedding.log')

class SingletonMetaclass(type):
    """metaclass定义"""
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass, self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance

class Embedding(metaclass=SingletonMetaclass):
    """
    词向量模型训练
    """
    def load_data(self):
        logger.info("load data")

        self.data = pd.concat([
            pd.read_csv(root_path + '/data/train_clean.csv', dtype=str, sep='\t'),
            pd.read_csv(root_path + '/data/dev_clean.csv', dtype=str, sep='\t'),
            pd.read_csv(root_path + '/data/test_clean.csv', dtype=str, sep='\t')
        ])

        self.stopWords = open(root_path + '/data/stopwords.txt').readlines()

        self.autoencoder = AutoEncoder()

        self.allWords = []        
        for sentence in self.data['text']:
            self.allWords.append(sentence.split(' '))

    def train(self):
        logger.info('train tfidf')
        cout_vect = TfidfVectorizer(stop_words=self.stopWords, 
                                    max_df=0.6, 
                                    ngram_range=(1, 2), 
                                    lowercase=False)
        self.tfidf_model = cout_vect.fit_transform(self.data.text)

        logger.info('train word2vec')
        self.w2v_model = models.Word2Vec(min_count=2,
                        window=5,
                        vector_size= config.embedding_size,
                        sample=6e-5,
                        alpha=0.007,
                        negative=15,
                        workers=4,
                        epochs=30,
                        max_vocab_size=config.max_vocab_size)

        self.w2v_model.build_vocab(self.allWords)
        self.w2v_model.train(self.allWords, 
                    total_examples=self.w2v_model.corpus_count, 
                    epochs=30, 
                    report_delay=1)

        logger.info('train fasttext')

        self.fast_model = models.FastText(self.allWords,
                                vector_size=config.embedding_size,  # 向量维度
                                window=3,    # 窗口大小
                                alpha=0.03,
                                min_count=2,  # 对字典进行截断，小于该数的则会被切掉,增大该值可以减少词表个数
                                epochs=15,    # 迭代次数
                                min_n=1,
                                max_n=3,
                                word_ngrams=1,
                                max_vocab_size=config.max_vocab_size)

        logger.info('train lda')

        self.id2word = gensim.corpora.Dictionary(self.allWords)
        corpus = [self.id2word.doc2bow(text) for text in self.allWords]

        self.LDAmodel = LdaMulticore(corpus=corpus,
                                     id2word=self.id2word,
                                     num_topics=30,
                                     workers=4,
                                     chunksize=4000,
                                     passes=7,
                                     alpha='asymmetric')

        logger.info('train autoencoder')
        self.autoencoder.train(self.data)

    def save_model(self):
        """
        保存训练好的模型
        """
        model_path = root_path + '/model/embedding/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        logger.info('save tfidf model')
        joblib.dump(self.tfidf_model, model_path + 'tfidf.model')

        logger.info('save word2vec model')
        self.w2v_model.save(model_path + 'word2vec.model')

        logger.info('save fasttext model')
        self.fast_model.save(model_path + 'fasttext.model')

        logger.info('save lda model')
        self.LDAmodel.save(model_path + '/lda.model')        

    def load_model(self):
        """
        加载保存训练好的模型
        """
        model_path = root_path + '/model/embedding/'

        logger.info('load tfidf model')
        self.tfidf_model = joblib.load(model_path + 'tfidf.model')

        logger.info('load w2v model')
        self.w2v_model = models.Word2Vec.load(model_path + 'word2vec.model')

        logger.info('load fast model')
        self.fast_model = models.FastText.load(model_path + 'fasttext.model')

        logger.info('load lda model')
        self.LDAmodel = self.autoencoder.load(model_path + 'lda.model')        

if __name__ == '__main__':
    embedding = Embedding()
    embedding.load_data()
    embedding.train()
    embedding.save_model()