import logging
import re
import numpy as np
import pandas as pd
import time
from datetime import timedelta
import torch
import jieba
import lightgbm as lgb
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score
from skopt import BayesSearchCV
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from src.utils import config

tqdm.pandas()

def get_time_dif(start_time):
    """
    获取时间间隔秒数
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def clean_str(string):
    """
    去除大小写英文数字等字符
    """
    string = re.sub(r"\s+", "", string)
    string = re.sub(r"[^\u4e00-\u9fa5^.^,^!^?^:^;^、^a-z^A-Z^0-9]", "", string)
    return string.strip()

def strQ2B(ustring):
    """
    全角转换半角
    """
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)

def sentence_embedding(sentence, w2v_model, method='mean', aggregate=True):
    """
    转换句子为word2vec词向量表示形式
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    method：聚合方法mean或max
    aggregate: 是否进行聚合
    """
    embedding = []
    for word in sentence.split():
        # 如果词在词典中
        if word in w2v_model.wv.index_to_key:
            embedding.append(w2v_model.wv.get_vector(word))
        #不在词典中就随机生成
        else:        
           embedding.append(np.random.randn(1, 300))

    if not aggregate:
        return np.array(embedding)

    if method == 'mean':
        embedding = np.mean(np.array[embedding], axis=0)
    elif method == 'max':
        embedding = np.nZ(np.array[embedding], axis=0)
    else:
        raise NotImplementedError

    embedding = embedding.reshape(300,)        
    return embedding


def padding(indice, max_length, pad_idx=0):
    """
    paddding操作，右侧补0
    """
    pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
    return torch.tensor(pad_indice)

def get_score(Train_label, Test_label, 
    Train_predict_label, Test_predict_label):
    """
    输出模型的准确率，召回率， f1_score
    """
    return accuracy_score(Train_label, Train_predict_label), \
            accuracy_score(Test_label, Test_predict_label),  \
            recall_score(Test_label, Test_predict_label, average='micro'), \
            f1_score(Test_label, Test_predict_label, average='weighted')

def query_cut(query):
    """
    对句子进行分词
    """
    return list(jieba.cut(query))

def bayes_parameter_opt_lgb(
        train_data,
        init_round=3,
        opt_round=5,
        n_folds=5,
        random_seed=6,
        n_estimators=10000,
        learning_rate=0.05):

    """
    lightgbm使用贝叶斯参数优化
    """

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth,
                 lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {
            'application': 'multiclass',
            'num_iterations': n_estimators,
            'learning_rate': learning_rate,
            'early_stopping_round': 100,
            'num_class': len([x.strip() for x in open(
                config.root_path +'/data/class.txt').readlines()]),
            'metric': 'multi_logloss'
        }
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight

        # 返回交叉验证结果
        cv_result = lgb.cv(params,
                           train_data,
                           nfold=n_folds,
                           seed=random_seed,
                           stratified=True,
                           verbose_eval=200)

        return max(cv_result['multi_logloss-mean'])

    # 贝叶斯参数优化搜索范围
    lgbBO = BayesianOptimization(lgb_eval, {    
        'num_leaves': (24, 45),
        'feature_fraction': (0.1, 0.9),
        'bagging_fraction': (0.8, 1),
        'max_depth': (5, 8.99),
        'lambda_l1': (0, 5),
        'lambda_l2': (0, 3),
        'min_split_gain': (0.001, 0.1),
        'min_child_weight': (5, 50)
    }, random_state=0)

    # 优化
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    # 返回最好的参数
    return lgbBO.max

def concate_data(train, train_tfidf, train_ae):
    # 将数据拼接到一起
    Data = pd.concat([
        train[[
            'labelIndex', 'length', 'capitals', 'caps_vs_length',
            'num_exclamation_marks', 'num_question_marks', 'num_punctuation',
            'num_symbols', 'num_words', 'num_unique_words', 'words_vs_unique',
            'nouns', 'adjectives', 'verbs', 'nouns_vs_length',
            'adjectives_vs_length', 'verbs_vs_length', 'nouns_vs_words',
            'adjectives_vs_words', 'verbs_vs_words', 'count_words_title',
            'mean_word_len', 'punct_percent'
        ]], train_tfidf, train_ae
    ] + [
        pd.DataFrame(
            train[i].tolist(),
            columns=[i + str(x) for x in range(train[i].iloc[0].shape[0])])
        for i in [
            'w2v_label_mean', 'w2v_label_max', 'w2v_mean', 'w2v_max',
            'w2v_win_2_mean', 'w2v_win_3_mean', 'w2v_win_4_mean',
            'w2v_win_2_max', 'w2v_win_3_max', 'w2v_win_4_max', 'res_embedding',
            'resnext_embedding', 'wide_embedding', 'bert_embedding', 'lda'
        ]
    ], axis=1).fillna(0.0)
    return Data


def format_data(data, max_features, maxlen, tokenizer=None, shuffle=False):
    '''
    max_features： 最大的特征的个数
    maxlen： 最大长度
    tokenizer： 分词器
    shuffle： 是否打乱顺序
    '''
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

    data['text'] = data['text'].apply(lambda x: str(x).lower())

    X = data['text']

    if not tokenizer:
        filters = "\"#$%&()*+./<=>@[\\]^_`{|}~\t\n"
        tokenizer = Tokenizer(num_words=max_features, filters=filters)
        tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)

    return X, tokenizer