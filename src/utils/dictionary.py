from collections import Counter
from src.utils import config

class Dictionary(object):
    def __init__(self,  
                max_vocab_size=config.max_vocab_size,   # 最大词典大小
                min_count=None,                         # 词的最小出现次数  
                start_end_tokens=False,                 # <SOS> <EOS>起始结束标志
                wordvec_mode=None,                      # 是否使用word2vec词向量
                embedding_size=config.embedding_size):  # 词向量维度
                
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.start_end_tokens = start_end_tokens
        self.embedding_size = embedding_size
        self.wordvec_mode = wordvec_mode        
        self.PAD_TOKEN = '<PAD>'

    def build_dictionary(self, data):
        self.voacab_words, self.word2idx, self.idx2word, self.idx2count = \
            self._build_dictionary(data)

        self.vocabulary_size = len(self.voacab_words)

        if self.wordvec_mode is None:
            self.embedding = None
        elif self.wordvec_mode == 'word2vec':
            self.embedding = self._load_word2vec()

    def indexer(self, word):
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']

    def _build_dictionary(self, data):

        # 初始化2个词
        vocab_words = [self.PAD_TOKEN, '<UNK>']
        vocab_size = 2

        # 起始结束token
        if self.start_end_tokens:
            vocab_words += ['<SOS>', '<EOS>']
            vocab_size += 2

        # 统计所有词出现的次数，返回字典 {word: 出现次数}
        counter = Counter([word for sentence in data for word in sentence.split()])

        # 只取最大词典数量
        if self.max_vocab_size:
            counter = {word: freq for word, freq in counter.most_common(
                self.max_vocab_size - vocab_size)}

        # 过滤掉低频词
        if self.min_count:
            counter = {word: freq for word, freq in counter.item() if freq >= self.min_count}

        # 所有的词
        vocab_words += list(sorted(counter.keys()))

        # 词出现的频次列表
        idx2count = [counter.get(word, 0) for word in vocab_words]

        # 词到索引的列表
        word2idx = {word: idx for idx, word in enumerate(vocab_words)}

        # 索引到词的列表
        idx2word = {idx: word for idx, word in enumerate(vocab_words)}
        
        return vocab_words, word2idx, idx2word, idx2count      
