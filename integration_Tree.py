from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # show data info
    data = load_iris()  # 加载 IRIS 数据集
    print('keys: \n', data.keys())  # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
    feature_names = data.get('feature_names')
    print('feature names: \n', data.get('feature_names'))  # 查看属性名称
    print('target names: \n', data.get('target_names'))  # 查看 label 名称
    x = data.get('data')  # 获取样本矩阵
    y = data.get('target')  # 获取与样本对应的 label 向量
    print(x.shape, y.shape)  # 查看样本数据
    # print(data.get('DESCR'))
    f = [y == 0, y == 1, y == 2]
    color = ['red', 'blue', 'green']
    fig, axes = plt.subplots(4, 4)  # 绘制四个属性两两之间的散点图
    for i, ax in enumerate(axes.flat):
        row = i // 4
        col = i % 4
        if row == col:
            ax.text(.1, .5, feature_names[row])
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        for k in range(3):
            ax.scatter(x[f[k], row], x[f[k], col], c=color[k], s=3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 设置间距
    plt.show()

    # 随机划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
    num_test = len(y_test)
    # 构建决策树
    clf = tree.DecisionTreeClassifier(criterion="entropy")  # 建立决策树对象
    clf.fit(x_train, y_train)  # 决策树拟合
    tree.plot_tree(clf)
    plt.show()
    # 预测
    y_test_pre = clf.predict(x_test)  # 利用拟合的决策树进行预测
    print('the predict values are', y_test_pre)  # 显示结果

    acc = sum(y_test_pre == y_test) / num_test
    print('the accuracy is', acc)  # 显示预测准确率


