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
import os

logger = create_logger(config.log_dir + 'sklearn_models_fasttext.log')

fasttext = models.FastText.load(
    root_path + '/model/embedding/fasttext.model')

print("fasttext词表的个数：{}".format(len(fasttext.wv.index_to_key)))

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

def sentence2vec(sentence):
    """
    转换句子的向量表示
    """
    vectors = []
    vectors = [fasttext.wv.get_vector(word) 
                for word in sentence 
                if word in fasttext.wv.index_to_key]    

    if len(vectors) > 0:
        return np.mean(np.array(vectors), axis=0)
    else:
        return np.zeros(config.embedding_size)

def generate_train_test_features():
    """
    生成训练数据集和测试数据集的向量化表示
    并且进行归一化处理，方便送入模型
    """
    min_max_scaller = preprocessing.MinMaxScaler()

    train_features = train_data['queryCutRMStopWord'].apply(sentence2vec)
    train_features = np.vstack(train_features)
    train_features = min_max_scaller.fit_transform(train_features)

    test_features = test_data['queryCutRMStopWord'].apply(sentence2vec)
    test_features = np.vstack(test_features)
    test_features = min_max_scaller.fit_transform(test_features)    

    train_label = train_data['labelIndex']
    test_label = test_data['labelIndex']

    return train_features, test_features, train_label, test_label

def train_and_test(train_features, test_features, train_label, test_label):

    embedding_type = "fasttext_embedding"
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

            # 预测错误的列表
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

                logger.info("预测错误的样本：{}, 预测标签: {}, 真实标签: {}".
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