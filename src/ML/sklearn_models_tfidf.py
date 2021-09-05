import numpy as np
import pandas as pd
from io import open
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import joblib
from src.utils import config
from src.utils.tools import create_logger
from src.utils.config import root_path
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os

logger = create_logger(config.log_dir + 'sklearn_models_tfidf.log')

train_data = pd.read_csv(root_path + '/data/train_clean.csv', sep='\t')
test_data = pd.read_csv(root_path + '/data/test_clean.csv', sep='\t')

# 标签去重
labels = train_data['label'].unique()

# 标签索引
label_index = list(range(len(labels)))

label_to_index = dict(zip(labels, label_index))
index_to_label = dict(zip(label_index, labels))

train_data['labelIndex'] = train_data['label'].map(label_to_index)
test_data['labelIndex'] = test_data['label'].map(label_to_index)

def sentence_cut(sentence):
    return sentence.split(' ')

train_data['queryCut'] = train_data['text'].apply(sentence_cut)
test_data['queryCut'] = test_data['text'].apply(sentence_cut)

stopwords = open(root_path + '/data/stopwords.txt').readlines()

def remove_stop_words(words):
    return [word for word in words if word not in stopwords]

train_data['queryCutRMStopWord'] = train_data['queryCut'].apply(remove_stop_words)
test_data['queryCutRMStopWord'] = test_data['queryCut'].apply(remove_stop_words)

allwords = [word for sentence in train_data['queryCutRMStopWord'] for word in sentence]

# 统计词频，返回字典 {单词 : 出现次数}
words_freq = Counter(allwords)

# 计算词频超过3的词列表
high_Words_freq = [word for word in words_freq.keys() if words_freq[word] > 3]

def remove_low_words_freq(sentence):
    return [word for word in sentence if word in high_Words_freq]

# train_data["queryFinal"] = train_data["queryCutRMStopWord"].apply(remove_low_words_freq)
# test_data["queryFinal"] = test_data["queryCutRMStopWord"].apply(remove_low_words_freq)

def generate_train_test_features():
    """
    生成训练数据集和测试数据集的向量化表示
    """
    train_text = [' '.join(word) for word in train_data['queryCutRMStopWord']]
    test_text = [' '.join(word) for word in test_data['queryCutRMStopWord']]

    vectorizer = TfidfVectorizer(stop_words=stopwords,  max_df=0.4, 
                                min_df=0.001, ngram_range=(1, 2))

    train_features = vectorizer.fit_transform(train_text)
    test_features = vectorizer.transform(test_text)

    train_label = train_data['labelIndex']
    test_label = test_data['labelIndex']

    return train_features, test_features, train_label, test_label

def train_and_test(train_features, test_features, train_label, test_label):

    embedding_type = "tfidf_embedding"
    models = [RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42),
        LogisticRegression(solver='liblinear', random_state=42),
        MultinomialNB(),
        SVC(),
        lgb.LGBMClassifier(objective='multiclass', n_jobs=10,
                           num_class=33, num_leaves=30, reg_alpha=10,
                           reg_lambda=200, max_depth=3, learning_rate=0.05,
                           n_estimators=2000, bagging_freq=1,
                           bagging_fraction=0.8, feature_fraction=0.8)]

    for model in models:
        model_name = model.__class__.__name__
        model.fit(train_features, train_label)

        print(f"使用的词嵌入类型：{embedding_type}，使用的模型：{model_name}")

        train_pred = model.predict(train_features)
        test_pred = model.predict(test_features)

        # 训练集准确率
        log = embedding_type + ' ' + model_name + ' ' + 'train accuracy: %s' \
            % metrics.accuracy_score(train_label, train_pred)
        print(log), logger.info(log)

         # 测试集准确率
        log = embedding_type + ' ' + model_name + ' ' + 'test accuracy: %s' \
            % metrics.accuracy_score(test_label, test_pred)
        print(log), logger.info(log)

        # 输出recall
        log = embedding_type + ' ' + model_name + ' ' + 'test recall: %s' \
            % metrics.recall_score(test_label, test_pred, average='micro')
        print(log), logger.info(log)

        # 输出地f1 score
        log = embedding_type + ' ' + model_name + ' ' + 'test f1_score: %s' \
            % metrics.f1_score(test_label, test_pred, average='weighted')
        print(log), logger.info(log)      

        # 输出精确率
        log = embedding_type + ' ' + model_name + ' ' + 'test precision_score: %s' \
            % metrics.precision_score(test_label, test_pred, average='micro')
        print(log), logger.info(log)

        # 显示预测错误的样本
        if config.show_error_predict:        

            # 预测错误的数量
            predict_error_list = np.argwhere(np.array(test_label - test_pred) != 0)
            error_dict = {}
            total_count = 0
            
            for k in range(len(predict_error_list)):
                # 不在预测错误的这个类别中就加入
                if int(test_pred[predict_error_list[k]]) not in error_dict.keys():
                    error_dict[int(test_pred[predict_error_list[k]])] = list(predict_error_list[k])
                    total_count += 1
                else:
                    # 更新预测错误的这个类别的计数
                    if len(error_dict[int(test_pred[predict_error_list[k]])]) < config.error_num_to_show:
                        error_dict[int(test_pred[predict_error_list[k]])].append(predict_error_list[k])
                        total_count += 1
                    else:
                        continue

                logger.info("预测错误的样本：{}, 预测标签：{}, 真实标签：{}".
                    format(np.array(test_data["queryCutRMStopWord"])[predict_error_list[k]],
                    index_to_label[int(test_pred[predict_error_list[k]])],
                    index_to_label[int(Test_label[predict_error_list[k]])]))                   

                # 每个类别都输出了5个预测错误的
                if total_count >= len(label_index) * config.error_num_to_show:
                    break        

        save_path = root_path + '/ML/saved_models/'
        if not os.path.exists(save_path): 
            os.mkdir(save_path)
        joblib.dump(model, save_path + embedding_type + "_" + model_name +'.pkl')        

if __name__ == '__main__':
    Train_features, Test_features, Train_label, Test_label = generate_train_test_features()
    train_and_test(Train_features, Test_features, Train_label, Test_label)