class config:
    # 语料库相关信息
    def __init__(self):
        self.vocab_path = './aclImdb/imdb.vocab'
        self.corpus_path = './aclImdb/train'
        self.test_path = './aclImdb/test'
        self.save_dict_path = './dict/dict.txt'


class LSTMConfig(config):
    # 模型相关信息，可以在此处对参数进行调整
    def __init__(self):
        super().__init__()
        self.lr = 0.005
        self.batch_size = 64
        self.epoch = 50
        self.embedding_dim = 100
        self.hidden_dim = 512
        self.p = 0.5
        self.model_path = './textClassifierModel/LSTM_SDPATTENTION-2.model'
