import torch
import os
import re
from torch.utils.data import Dataset


class loadCorpus:
    def __init__(self, vocab_path, corpus_path):
        print('loading corpus...')
        self.vocab_path = vocab_path
        self.corpus_path = corpus_path
        self.word2index = {}
        self.index2word = {}

        self.word2index['<pad>'] = 0
        self.index2word[0] = '<pad>'
        self.word2index['<num>'] = 1
        self.index2word[1] = '<num>'
        self.word2index['<unk>'] = 2
        self.index2word[2] = ['<unk>']

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = f.read().split('\n')
        f.close()

        self.vocab_num = 3
        for word in self.vocab:
            self.word2index[word] = self.vocab_num
            self.index2word[self.vocab_num] = word
            self.vocab_num += 1

        self.negtext_path = os.path.join(corpus_path, 'neg')
        self.postext_path = os.path.join(corpus_path, 'pos')
        self.text_list = []  # 训练语料文本的list，每个元素是一个完整的句子
        self.label_list = []  # 一句话的标签
        self.index_list = []  # 对训练语料文本进行索引的list，每个元素是一个完整的句子

        self.unk = 0
        self.k = 0
        self.unk_list = []

        self.add_text(self.negtext_path, 0)
        self.add_text(self.postext_path, 1)
        print('done!')

    def filter(self, text):
        # 数据清洗，将数据集中的标点符号等字符进行替换
        temptext = text
        temptext = text.lower().replace('`', '\'').replace('´', '\'').replace('’', '\''). \
            replace('«', ' ').replace('»', ' ').replace('”', ' ').replace('~', ' '). \
            replace(',', ' ').replace('.', ' ').replace('[', ' ').replace(']', ' '). \
            replace('<', ' ').replace('>', ' ').replace(':', ' ').replace('"', ' '). \
            replace('(', ' ').replace(')', ' ').replace('&', ' and ').replace(';', ' '). \
            replace('!', ' ').replace('?', ' ').replace('+', ' ').replace('*', ' '). \
            replace('{', ' ').replace('}', ' ').replace('_', ' ').replace('=', ' '). \
            replace('%', '').replace('#', '').replace('$', '').replace('£', '')

        temptext = re.sub('br /', ' ', temptext)
        temptext = re.sub('[0-9]+', ' <num> ', temptext)
        temptext = re.sub('\'s', ' ', temptext)
        temptext = re.sub('--+', ' - ', temptext)
        temptext = re.sub(' [-\']+', ' ', temptext)
        temptext = re.sub('[-\']+ ', ' ', temptext)

        temptext = temptext.replace('\x95', ' ').replace('\x96', ' ').replace('\x97', ' '). \
            replace('\x84', ' ').replace('\x91', ' ')

        temptext = temptext.replace('\\', ' ').replace('/', ' ').split()
        return temptext

    def add_text(self, path, label):
        # 根据路径读取所有文档中的内容
        for dirpath, dirnames, filenames in os.walk(path):
            for name in filenames:
                temptext = []
                with open(os.path.join(dirpath, name), 'r', encoding='utf-8') as f:
                    temptext = self.filter(f.read())
                f.close()

                tempindex = []
                self.label_list.append(torch.tensor([label]))
                for word in temptext:

                    if self.word2index.get(word) is not None:
                        tempindex.append(self.word2index[word])

                        self.k += 1
                    else:
                        tempindex.append(self.word2index['<unk>'])
                        self.unk_list.append(word)
                        self.unk += 1
                tempindex = torch.tensor(tempindex)
                self.text_list.append(temptext)
                self.index_list.append(tempindex)


class textDataset(Dataset):
    # 根据loadCorpus中的数据和标签，构建一个dataset
    def __init__(self, data_list, label_list):
        self.data = data_list
        self.label = label_list

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    lc = loadCorpus('./aclImdb/imdb.vocab', './aclImdb/train')
    ds = textDataset(lc.index_list, lc.label_list)
    print(lc.unk, lc.k)
    print(lc.unk_list[0:1000])


