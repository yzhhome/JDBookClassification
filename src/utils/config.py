import os
import torch

# 当前文件所在的路径
cur_path = os.path.abspath(os.path.dirname(__file__))

# 项目所在的根目录路径
root_path = os.path.split(os.path.split(cur_path)[0])[0]

# 训练集文件路径
train_file = root_path + '/data/train_clean.csv'

# 验证集文件路径
dev_file = root_path + '/data/dev_clean.csv'

# 测试集文件路径
test_file = root_path + '/data/test_clean.csv'

# 停用词文件路径
stopWords_file = root_path + '/data/stopwords.txt'

# 词典文件路径
vocab_file = root_path + '/model/vocab.bin'

# 是否显示预测错误的样本
show_error_predict = False

# 每个类别显示的错误样本数量
error_num_to_show = 5

# 日志文件保存目录
log_dir = root_path + '/logs/'

# cuda是否可用
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 最大序列长度
max_seq_length = 400

# 词向量维度
embedding_size = 300

# 词典最大大小
max_vocab_size = 50000

# True为使用词向量，False为使用字向量
use_word = True