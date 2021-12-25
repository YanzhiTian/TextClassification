import config
import readdict
import torch
import re
from LSTMClassifier import LSTM
import math
import tkinter as tk
from tkinter import filedialog
from tkinter import StringVar


def filter(text):
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


if __name__ == '__main__':
    cfg = config.LSTMConfig()
    word_dict = readdict.ReadDict(cfg)
    model = torch.load(cfg.model_path, map_location=torch.device('cpu'))
    model.eval()  # 关闭模型中的dropout层
    # GUI界面
    top = tk.Tk()
    top.title('文本分类')
    top.geometry('520x370+100+100')

    result_text = StringVar()
    result_text.set('分类结果:')
    score_text = StringVar()
    score_text.set('置信度：')
    text1 = tk.Text(top, width=70, height=20)
    text1.pack()
    text1.place(x=10, y=25)
    label1 = tk.Label(top, text='待分类文本：')
    label1.pack()
    label1.place(x=10, y=0)
    label2 = tk.Label(top, textvariable=score_text)
    label2.pack()
    label2.place(x=25, y=330)
    label3 = tk.Label(top, textvariable=result_text)
    label3.pack()
    label3.place(x=25, y=300)

    def predict(text):
        clean_text = filter(text)
        index = []
        for word in clean_text:
            if word_dict.word2index.get(word) is not None:
                index.append(word_dict.word2index[word])
            else:
                index.append(word_dict.word2index['<unk>'])
        index = torch.tensor(index)
        index = index.unsqueeze(0)
        pred = model(index)
        pred = pred.squeeze(0)
        return pred

    def cmd1():
        text = text1.get(0.0, "end")
        pred = predict(text)
        result_text.set('分类结果：positive' if torch.argmax(pred) == 1 else '分类结果：negative')
        score_text.set('置信度：' + str(math.exp(pred[torch.argmax(pred)]) / (math.exp(pred[0]) + math.exp(pred[1]))*100) + '%')

    def cmd2():
        file_path = filedialog.askopenfilename(title='选择文件', initialdir='./', filetypes=[('文本文件', '*.txt')])
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            pred = predict(text)
            text1.delete("1.0", "end")
            text1.insert("end", text)
            result_text.set('分类结果：positive' if torch.argmax(pred) == 1 else '分类结果：negative')
            score_text.set('置信度：' + str(math.exp(pred[torch.argmax(pred)])/(math.exp(pred[0]) + math.exp(pred[1]))*100) + '%')
        f.close()

    button1 = tk.Button(top, text='开始分类', command=cmd1, width=15, height=1)
    button1.pack()
    button1.place(x=375, y=300)
    button2 = tk.Button(top, text='选择文本进行分类', command=cmd2, width=15, height=1)
    button2.pack()
    button2.place(x=375, y=330)

    top.mainloop()
