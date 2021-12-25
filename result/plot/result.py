import matplotlib.pyplot as plt
import numpy as np

neg_acc = np.array([
    0.7828,
    0.8004,
    0.73912,
    0.84888
])
pos_acc = np.array([
    0.72752,
    0.71776,
    0.8376,
    0.81128
])
pos_p = np.array([
    0.7700906088576509,
    0.7824191157233802,
    0.762508193139611,
    0.842975893599335
])
pos_r = np.array([
    0.72752,
    0.71776,
    0.8376,
    0.81128
])
pos_f1 = np.array([
    0.74820025504957,
    0.7486961238369425,
    0.7982920971369754,
    0.8268242967794538
])
neg_p = np.array([
    0.7417936471836859,
    0.7393039237419641,
    0.8198597923506966,
    0.8181187355435621
])
neg_r= np.array([
    0.7828,
    0.8004,
    0.73912,
    0.84888
])
neg_f1 = np.array([
    0.7617453582966797,
    0.768639803326547,
    0.7773991333249191,
    0.8332155477031803
])
x = np.array([1,3,5,7])
plt.title('Accuracy')
plt.plot(x, neg_acc, label='neg_Accuracy')
plt.plot(x, pos_acc, label='pos_Accuracy')
plt.legend()
plt.xlim(0,8)
plt.ylim(0.5,1)
plt.xticks([1,3,5,7], ['LSTM','LSTM+attention','LSTM+\nscaled dot-product attention','lr=1e-5'])
plt.show()
plt.title('Positive')
plt.plot(x, pos_p, label='pos_Precision')
plt.plot(x, pos_r, label='pos_Recall')
plt.plot(x, pos_f1, label='pos_F1')
plt.legend()
plt.xlim(0,8)
plt.ylim(0.5,1)
plt.xticks([1,3,5,7], ['LSTM','LSTM+attention','LSTM+\nscaled dot-product attention','lr=1e-5'])
plt.show()
plt.title('Negative')
plt.plot(x, neg_p, label='neg_Precision')
plt.plot(x, neg_r, label='neg_Recall')
plt.plot(x, neg_f1, label='neg_F1')
plt.legend()
plt.xlim(0,8)
plt.ylim(0.5,1)
plt.xticks([1,3,5,7], ['LSTM','LSTM+attention','LSTM+\nscaled dot-product attention','lr=1e-5'])
plt.show()