import pandas as pd
from tqdm import tqdm
import fastText
from src.utils.config import root_path
from src.utils.config import train_file
from src.utils.config import test_file
from src.utils.tools import create_logger

logger = create_logger(root_path + '/logs/fasttext.log')

class FastText(object):
    """
    使用单独的fasttext模型，不使用gensim.FastText
    """

    def __init__(self, model_path=None):
        if model_path != None:
            self.fast_model = fastText.load_model(model_path)
        else:
            self.model_train_file = root_path + '/data/fast_train.csv'
            self.model_test_file = root_path + '/data/fast_test.csv'

            self.train_raw_data = pd.read_csv(train_file, sep='\t')
            self.test_raw_data = pd.read_csv(test_file, sep='\t')

            print('process train data')
            self.data_process(self.train_raw_data, self.model_train_file)

            print('process test data')
            self.data_process(self.test_raw_data, self.model_test_file)    

    def data_process(self, raw_data, model_data_file):
        """
        转换成fasttext模型所需的数据格式
        """
        with open(model_data_file, 'w') as f:
            for index, row in tqdm(raw_data.iterrows()):
                outline = row['text'] + "\t__label__" + \
                    str(int(row['category_id'])) + "\n"
                f.write(outline)

    def train(self):
        self.classifier = fastText.train_supervised(
            self.model_train_file,     # 训练数据集
            label="__label__",         # 标签字段
            dim=50,                    # 词向量维度
            epoch=30,                  # 训练的轮数
            lr=0.1,                    # 学习率 
            wordNgrams=2,              # 使用bi-gram
            loss='softmax',            # 使用softmax计算损失
            thread=50,                 # 线程数
            verbose=True)              # 显示日志

        self.classifier.save_model(root_path + '/model/fasttext_alone.model')

    def test(self):
        train_pred = self.classifier.test(self.model_train_file)
        test_pred = self.classifier.test(self.model_test_file)

        # 返回精确率和召回率
        print("train accuracy:", train_pred[1], "recall:", train_pred[2])   
        print("test accuracy:", test_pred[1], "recall:", test_pred[2])   

if __name__ == '__main__':
    fast_model = FastText()
    fast_model.train()
    fast_model.test()           