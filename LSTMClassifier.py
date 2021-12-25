import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torch.optim as optim
import loadCorpus
import config
import math


class LSTM(nn.Module):
    def __init__(self, n_vocab, embedding_dim=100, hidden_dim=512, p=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_vocab, self.embedding_dim)  # 设置词嵌入embedding层
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=1, bidirectional=True, batch_first=True)  # 设置双向LSTM层
        self.dropout = nn.Dropout(p=p)
        self.linear = nn.Linear(self.hidden_dim * 2, 2)  # 在LSTM层后增加全连接层
        self.w_q = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.w_k = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.w_v = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)

    def attention(self, _hn, _lstmout, x):
        q = _hn.view(len(x), 1, self.hidden_dim * 2)  # 将h_n设为q向量，shape:[batch, 1, hidden_dim*2]
        K = _lstmout.reshape(len(x), -1, self.hidden_dim * 2)  # 将LSTM的output设为K矩阵，shape:[batch, seq_len, hidden_dim*2]
        K_T = _lstmout.reshape(len(x), self.hidden_dim * 2, -1)  # K的转置矩阵
        att_weight = f.softmax(torch.matmul(q, K_T), 2)  # 相乘后shape:[batch, 1, seq_len]
        att_output = torch.matmul(att_weight, K)  # 相乘后shape:[batch, 1, hidden_dim*2]
        return att_output

    def self_attention(self, Q, K, V):
        att_score = torch.matmul(Q, torch.transpose(K, 1, 2)) / math.sqrt(2 * self.hidden_dim)
        att_weight = f.softmax(att_score, 2)
        att_output = torch.matmul(att_weight, V)  # shape:[batch, seq, 2*hidden_dim]
        return att_output.mean(1)  # shape:[batch, 1, 2*hidden_dim]

    def forward(self, x):
        embedding_out = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedding_out)
        # hn.shape:[2, batch, hidden_dim], out.shape:[batch, len(sequence), 2*hidden_dim]

        # lstm_out = torch.mean(lstm_out, dim=1)  # 仅用于LSTM模型
        dropout_out = self.dropout(lstm_out)
        # attention_out = self.attention(hn, lstm_out, x)  # 仅用于LSTM+ATTENTION模型

        # transformer原论文中使用的attention(scaled dot-product attention)方法 #
        Q = self.w_q(lstm_out)
        K = self.w_k(lstm_out)
        V = self.w_v(lstm_out)
        attention_out = self.self_attention(Q, K, V)
        # ---------- #
        fc = self.linear(attention_out)  # 用于LSTM+ATTENTION模型，LSTM+SDPATTENTION(scaled dot-product attention)模型
        # fc = self.linear(dropout_out)  # 仅用于LSTM模型
        return fc


class MyCollate:
    # dataloader中的collate函数，目的是对一个batch中的句子做padding
    def __init__(self, pad_value):
        self.padding = pad_value

    def __call__(self, batch_data):
        x_ = []
        y_ = []
        for x, y in batch_data:
            x_.append(x)
            y_.append(y)
        x_ = nn.utils.rnn.pad_sequence(x_, batch_first=True, padding_value=self.padding)
        y_ = torch.tensor(y_)
        return x_, y_


if __name__ == '__main__':
    cfg = config.LSTMConfig()
    lc = loadCorpus.loadCorpus(cfg.vocab_path, cfg.corpus_path)
    ds = loadCorpus.textDataset(lc.index_list, lc.label_list)
    dl = DataLoader(dataset=ds, batch_size=cfg.batch_size, collate_fn=MyCollate(0), shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LSTM(lc.vocab_num, cfg.embedding_dim, cfg.hidden_dim, cfg.p).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)  # 使用Adam优化器
    loss_function = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    for epoch in range(cfg.epoch):
        running_loss = 0.
        correct = 0
        total = 0
        for batch, (x, y) in enumerate(dl):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            x.type(torch.LongTensor)
            y.type(torch.LongTensor)
            pred = model(x).to(device)

            pred = pred.view(-1, 2)
            y = y.view(-1)

            correct += (pred.argmax(-1) == y).sum().item()  # 统计预测正确的个数
            total += len(y)

            loss = loss_function(pred, y).to(device)
            running_loss += float(loss)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                print("loss=", float(loss))
        print('epoch:', epoch + 1, ' loss:', running_loss / len(dl), 'acc:', correct / total)
        torch.save(model, cfg.model_path)
