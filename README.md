北京理工大学自然语言理解初步大作业2--英文文本分类

作者：田炎智 邮箱：yz_tian113@163.com

-aclImdb文件夹中包含训练、测试使用的数据集(请于http://ai.stanford.edu/~amaas/data/sentiment/下载)

-dict文件夹中包含保存的词典

-result包含所有训练数据和结果

​	-pic包含运行数据和结果的图片

​	-plot包含绘制图片的程序

-textClassifierModel包含训练的所有可用模型(模型请见项目的release)

-app.py 可运行程序入口

-calculatePRF.py 计算模型预测结果的P,R,F1值

-config.py 包含模型参数的设定、数据集的路径

-loadCorpus.py 读取数据集

-LSTMClassifier.py 训练模型

-LSTMTest.py 测试模型

-readdict.py 读取已保存的词典

-savedict.py 保存读取数据集获得的词典
