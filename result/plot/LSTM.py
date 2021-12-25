import matplotlib.pyplot as plt
import numpy as np
loss = np.array([0.5582175486532929,
                0.2606749696004421,
                0.1759311116736411,
                0.2471736286245191,
                0.08938872561458965,
                0.05684186551211368,
                0.04072458962576411,
                0.034159498573045656,
                0.018656614731968668,
                0.023638439503402504,
                0.033454769154024476])
acc = np.array([0.7078,
                0.90348, 
                0.93596,
                0.91216,
                0.97232,
                0.98392,
                0.98888,
                0.99192,
                0.99556,
                0.99348,
                0.99072])
epoch = range(1,12)
plt.title('LSTM')
plt.xlabel('epoch')
plt.plot(epoch, loss, label='loss')
plt.plot(epoch, acc, label='accuracy')
plt.legend()
plt.show()