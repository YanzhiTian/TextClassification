import torch
import config
import loadCorpus
import os
from LSTMClassifier import LSTM


def Test(label, lc):
    if label == 'pos':
        test_path = os.path.join(cfg.test_path, 'pos')
        num_label = 1
    elif label == 'neg':
        test_path = os.path.join(cfg.test_path, 'neg')
        num_label = 0
    total_num = 0
    correct_num = 0

    for dirpath, dirnames, filenames in os.walk(test_path):
        for name in filenames:
            with open(os.path.join(dirpath, name), 'r', encoding='utf-8') as f:
                text = lc.filter(f.read())
            f.close()
            index = []
            for word in text:
                if lc.word2index.get(word) is not None:
                    index.append(lc.word2index[word])
                else:
                    index.append(lc.word2index['<unk>'])
            index = torch.tensor(index)
            index = index.unsqueeze(0)
            pred = model(index)
            total_num += 1
            if torch.argmax(pred) == num_label:
                correct_num += 1
            print(name, ':', 'positive' if torch.argmax(pred) == 1 else 'negative', 'score: ', pred)
    return total_num, correct_num


if __name__ == '__main__':
    # 测试模型的程序
    cfg = config.LSTMConfig()
    lc = loadCorpus.loadCorpus(cfg.vocab_path, cfg.corpus_path)
    model = torch.load(cfg.model_path, map_location=torch.device('cpu'))
    model.eval()  # 关闭模型中的dropout层

    neg_total, neg_correct = Test('neg', lc)
    pos_total, pos_correct = Test('pos', lc)

    print('neg: total: ', neg_total, ' correct: ', neg_correct, ' accuracy: ', neg_correct / neg_total)
    print('pos: total: ', pos_total, ' correct: ', pos_correct, ' accuracy: ', pos_correct / pos_total)
