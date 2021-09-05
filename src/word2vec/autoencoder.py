from keras.layers import Input, Dense, Bidirectional, Embedding, LSTM
from keras.layers import GlobalMaxPool1D
from keras.models import Model
from keras import regularizers
import joblib
import os
from src.utils.config import root_path
from src.utils.tools import format_data

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

class AutoEncoder(object):
    def __init__(self, max_features=500, max_len=200):

        self.max_len = max_len
        self.max_features = max_features
        self.init_model()

    def init_model(self):

        # encoder输入的句子
        input = Input(shape=(self.max_len, ))

        # 生成句子的embedding
        encoder = Embedding(self.max_features, 50)(input)

        # encoder 第一层双向lstm
        encoder = Bidirectional(LSTM(units=75, return_sequences=True))(encoder)

        # encoder 第二层双向lstm
        encoder = Bidirectional(LSTM(units=25, return_sequences=True, 
            activity_regularizer=regularizers.l1(l1=10e-5)))(encoder)
        
        # encoder输出
        encoder_output = Dense(units=self.max_features)(encoder)

        # decoder 双向lstm
        decoder = Bidirectional(LSTM(units=75, return_sequences=True))(encoder_output)

        # pooling
        decoder = GlobalMaxPool1D()(decoder)

        # 全连接层改变维度
        decoder = Dense(units=50, activation='relu')(decoder)
        decoder = Dense(units=self.max_features)(decoder)

        # 编译模型
        self.model = Model(inputs=input, outputs=decoder)

        self.model.compile(loss='mean_squared_error',
                            optimizer='adam',
                            metrics=['accuracy'])
        self.encoder = Model(inputs=input, outputs=encoder_output)

    def train(self, data, epochs=1):
        self.X, self.tokenizer = format_data(data, 
                                            self.max_features, 
                                            self.max_len,
                                            shuffle=True)
        self.model.fit(self.X, 
                        self.X,
                        epochs=epochs,
                        verbose=1)

    def save(self):
        save_path = root_path + '/model/embedding/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        joblib.dump(self.tokenizer, save_path + 'tokenizer.model')
        self.model.save_weights(save_path + 'autoencoder.model')
        self.encoder.save_weights(save_path + 'autoencoder_encoder.model')

    def load(self):
        load_path = root_path + '/model/embedding/'
        self.tokenizer = joblib.load(load_path + 'tokenizer.model')
        self.model.load_weights(load_path + 'autoencoder.model')
        self.encoder.load_weights(load_path + 'autoencoder_encoder.model')