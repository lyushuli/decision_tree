import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    x = load_iris().data
    y = load_iris().target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # 将数据集按照 训练集比测试集为8：2的比例随机拆分数据集

    clf = GaussianNB(var_smoothing=1e-8)
    clf.fit(x_train, y_train)  # 带入训练集训练模型
    num_test = len(y_test)
    # 预测
    y_test_pre = clf.predict(x_test)  # 利用拟合的贝叶斯进行预测
    print('the predict values are', y_test_pre)  # 显示结果
    acc = sum(y_test_pre == y_test) / num_test
    print('the accuracy is', acc)  # 显示预测准确率

    # 数据扩大
    Accuracy = []
    for j in range(60):
        x = load_iris().data
        y = load_iris().target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(80-j)/100)
        clf = GaussianNB(var_smoothing=1e-8)# 拟合模型
        clf.fit(x_train, y_train)  # 训练模型
        y_test_pre = clf.predict(x_test)  # 利用拟合的贝叶斯进行预测

        num_test = len(y_test)
        acc = sum(y_test_pre == y_test) / num_test
        Accuracy.append(acc)  # 计算准确率

    plt.figure()
    plt.plot(list(range(len(Accuracy))), Accuracy)
    plt.xlabel("data size")
    plt.ylabel("accuracy")
    plt.show()
    print()

    print('the final accuracy is', Accuracy[-1])  # 显示预测准确率
