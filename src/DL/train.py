import time
import torch
import numpy as np
import pandas as pd
import os
import joblib
from torch.utils.data import DataLoader
from importlib import import_module
from src.utils import config
from src.utils.tools import create_logger
from src.utils.dictionary import Dictionary
from train_helper import train, init_network
from src.utils.mydataset import MyDataset, batch_padding
import torchsnooper
torchsnooper.snoop()

logger = create_logger(config.root_path + '/logs/dl_models_train.log')

normal_models = ['CNN', 'DPCNN', 'RCNN', 'RNN_ATT', 'RNN', 'TRANSFORMER']
bert_models = ['bert', 'roberta', 'xlnet']

if __name__ == '__main__':
    for model_name in normal_models:
        logger.info('Current train model ' + model_name)

        module = import_module('models.' + model_name)
        module_config = module.Config(dataset=config.root_path)

        # 设置固定随机种子，让模型可复现
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  

        # 之前构建过词典直接加载
        if os.path.exists(config.vocab_file) and (os.path.getsize(config.vocab_file) > 1000):
            logger.info('load exist dictionary ...')
            dictionary = joblib.load(config.vocab_file)            
        else:
            logger.info('Building dictionary ...')

            data = pd.concat([pd.read_csv(config.train_file, sep='\t'),
                pd.read_csv(config.dev_file, dtype=str, sep='\t')])

            # 所有句子组成的列表
            if config.use_word:
                datalist = data['text'].values.tolist()
            # 所有字组成的列表
            else:
                datalist = data['text'].apply(lambda x: " ".join("".join(x.split())))

            # 构建词典
            dictionary = Dictionary()
            dictionary.build_dictionary(datalist)

            # 删除数据释放占用的空间
            del datalist

            # 保存词典
            joblib.dump(dictionary, config.vocab_file)

        # bert, roberta, xlnet模型使用自己的tokenizer
        if not model_name.isupper():
            tokenizer = module_config.tokenizer 

        # CNN, DPCNN, RCNN, RNN_ATT, RNN, TRANSFORMER 模型使用dictionary
        else:
            tokenizer = None

        logger.info('Generate Dataset and Dataloader...')
            
        # 训练集Dataset和DataLoader
        train_dataset = MyDataset(config.train_file, dictionary, config.max_seq_length, 
            tokenizer=tokenizer, use_word=config.use_word)

        train_dataloader = DataLoader(train_dataset, batch_size=module_config.batch_size, 
            shuffle=True, drop_last=False, collate_fn=batch_padding)
        
        # 验证集Dataset和DataLoader
        dev_dataset = MyDataset(config.dev_file, dictionary, config.max_seq_length,  
            tokenizer=tokenizer, use_word=config.use_word)

        dev_dataloader = DataLoader(dev_dataset, batch_size=module_config.batch_size, 
            shuffle=True, drop_last=False, collate_fn=batch_padding)
        
        # 测试集Dataset和DataLoader
        test_dataset = MyDataset(config.test_file, dictionary, config.max_seq_length,  
            tokenizer=tokenizer, use_word=config.use_word)

        test_dataloader = DataLoader(test_dataset, batch_size=module_config.batch_size, 
            shuffle=True, drop_last=False, collate_fn=batch_padding)

        # 获取当前导入的模型
        model = module.Model(module_config).to(config.device)

        if model_name != 'TRANSFORMER':
            init_network(model)

        print(model.parameters)

        logger.info('Start train model ' + model_name + '...')
        train(module_config, model, train_dataloader, dev_dataloader, test_dataloader)