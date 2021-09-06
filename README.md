## 京东AI NLP高阶实训营-京东图书分类项目

### 代码结构介绍
`data`: 数据存放目录

`src` : 核心代码部分

`app.py` : 代码部署部分

`src/utils/dataset.py` : 主要用于深度学习的数据处理

`src/utils/mlData.py` : 主要用于机器学习的数据处理

`src/DL/` : 包含各类深度学习模型， 运行主入口为`src/DL/train.py`

`src/ML/` : 包含各类机器学习模型， 运行主入口为`src/ML/main.py`

`src/utils/` : 包含配置文件，特征工程函数，以及通用函数

`src/word2vec/` : 包含各类embedding的训练，保存加载