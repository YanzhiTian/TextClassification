import config


class ReadDict:
    def __init__(self, cfg):
        self.cnt = 0
        self.word2index = {}
        with open(cfg.save_dict_path, 'r', encoding='utf-8') as f:
            words = f.read().strip()
            word_list = words.split('\n')
            for word in word_list:
                self.word2index[word] = self.cnt
                self.cnt += 1
        f.close()
        self.vocab_num = self.cnt


if __name__ == '__main__':
    # 首先使用savedict.py存储词典，之后将词典内容存入ReadDict类中，方便直接使用
    cfg = config.config()
    rd = ReadDict(cfg)
    print(rd.vocab_num)
