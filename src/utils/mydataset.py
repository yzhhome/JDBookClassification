import pandas as pd
import torch
import json
from torch.utils.data import Dataset
from src.utils import config

class MyDataset(Dataset):
    """
    继承自Dataset用来读取数据集，并转换成向量化表示
    """
    def __init__(self, file_path,           # 数据集文件路径
                        dictionary=None,    # 构建好的词典
                        max_length=128,     # 句子最大长度
                        tokenizer=None,     # bert的tokenizer
                        use_word=True):    # 是否使用词进行编码，False为使用字向量

        super(MyDataset, self).__init__()

        self.data = pd.read_csv(file_path, sep='\t').dropna()

        # 读取标签和id对
        with open(config.root_path + '/data/label2id.json', 'r') as f:
            self.label2id = json.load(f)
        
        self.data['category_id'] = self.data['label'].map(self.label2id)

        # 使用字向量
        if not use_word:
            self.data['text'] = self.data['text'].apply(lambda x: " ".join("".join(x.split())))

        if tokenizer is not None:
            self.model_name = 'bert'
            self.tokenizer = tokenizer
        else:
            self.model_name = 'normal'
            self.tokenizer = dictionary

        self.max_length = max_length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data.iloc[index]
        text = data['text']
        cat_id = data['category_id']
        if cat_id == cat_id:
            labels = int(cat_id)
        else:
            labels = 0
        
        if 'bert' in self.model_name:
            # 转换句向量编码
            text_dict = self.tokenizer.encode_plus(text, 
                            add_special_tokens=True,      # 自动添加[CLS]和[SEP]
                            max_length=self.max_length,   # Padding操作
                            ad_to_max_length=True,
                            return_attention_mask=True,   # 需要返回attenion_mask
                            return_tensors='pt')          # 返回pytorch tensor

            input_ids =  text_dict['input_ids']
            attention_mask =  text_dict['attention_mask']
            token_type_ids = text_dict['token_type_ids']                
        else:
            text = text.split()

            # 小于max_lenth补0，超出的部分截断
            text = text + [0] * max(0, self.max_length - len(text)) \
                if len(text) < self.max_length else text[0:self.max_length]
            
            # 返回句子中所有词的word2id表示
            input_ids = [self.tokenizer.indexer(word) for word in text]
            attention_mask = [0] * self.max_length
            token_type_ids = [0] * self.max_length

        output = {"token_ids": input_ids, 
                    'attention_mask': attention_mask, 
                    "token_type_ids": token_type_ids,
                     "labels": labels}
        return output

def batch_padding(batch):
    """
    动态padding， batch为一部分sample
    """
    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        token_type_ids右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])

    token_type_ids = [data["token_type_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]
    labels = torch.tensor([data["labels"] for data in batch])

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    attention_mask_padded = padding(attention_mask, max_length)

    return token_ids_padded, attention_mask_padded, token_type_ids_padded, labels