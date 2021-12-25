import numpy as np


def cal(M):
    pos_p = M[1, 1] / (M[1, 1] + M[0, 1])
    pos_r = M[1, 1] / (M[1, 1] + M[1, 0])
    neg_p = M[0, 0] / (M[0, 0] + M[1, 0])
    neg_r = M[0, 0] / (M[0, 0] + M[0, 1])
    print('pos_p: ', pos_p)
    print('pos_r: ', pos_r)
    print('pos_f1: ', 2 * pos_r * pos_p / (pos_p + pos_r))
    print('neg_p: ', neg_p)
    print('neg_r: ', neg_r)
    print('neg_f1: ', 2 * neg_r * neg_p / (neg_p + neg_r))


if __name__ == '__main__':
    '''
    用于计算准确率 召回率的混淆矩阵
    |      | 预测为0| 预测为1|
    |实际为0|   TN  |   FP  |
    |实际为1|   FN  |   TP  |
    '''
    print('LSTM: ')
    M = np.array([
        [9785., 2715.],
        [3406., 9094.]
    ])
    cal(M)
    print('LSTM+ATTENTION: ')
    M = np.array([
        [10005., 2495.],
        [3528., 8972.]
    ])
    cal(M)
    print('LSTM+SDPATTENTION: ')
    M = np.array([
        [9239., 3261.],
        [2030., 10470.]
    ])
    cal(M)
    print('LSTM+SDPATTENTION-2: ')
    M = np.array([
        [10611., 1889.],
        [2359., 10141.]
    ])
    cal(M)
