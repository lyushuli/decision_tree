import collections
import random
from math import log
import pandas as pd
import operator
import treePlotter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree


# 数据处理
# 获取数据 划分数据集
def get_dataset(test_size):
    dataset = pd.read_csv("iris.data").values.tolist()
    random.shuffle(dataset)
    # 特征值列表
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=2)
    feature_names = load_iris().get('feature_names')

    return train_dataset, test_dataset, feature_names


def entroy_calc(dataset):
    # 计算出数据集的总数
    num_data = len(dataset)
    # 创建字典用来统计标签数
    labelcounts = collections.defaultdict(int)

    # 统计标签数
    for data in dataset:
        label = data[-1]
        labelcounts[label] += 1
    # 信息熵初始化
    entroy = 0.0
    for key in labelcounts:
        # 计算分类标签占总标签的比例
        prob = float(labelcounts[key]) / num_data
        # 求信息熵
        entroy -= prob * log(prob, 2)

    return entroy


# 根据特征划分数据集
# feature为特征种类
# value为划分值
def split_part_dataset(dataset, feature, value):
    # 用来保存不大于划分值的集合
    low_dataset = []
    # 用来保存大于划分值的集合
    high_dataset = []
    # 进行划分，保留该特征值
    for data in dataset:
        if data[feature] <= value:
            low_dataset.append(data)
        else:
            high_dataset.append(data)

    return low_dataset, high_dataset


# 对给定的特征信息增益划分点计算
# i 特征值下标
# father_entropy 父节点信息熵，用于计算信息增益
def info_gain_calc(dataset, feature, father_entropy):
    # 记录最大的信息增益
    max_info_gain = 0.0
    # 最好的划分点
    best_dot = 0

    # 得到当前特征值列表
    feature_list = [data[feature] for data in dataset]
    # 得到分类列表
    class_list = [data[-1] for data in dataset]
    dict_list = dict(zip(feature_list, class_list))

    # 特征值从小到大排序
    sorted_feature_list = sorted(dict_list.items(), key=operator.itemgetter(0))

    # 计算连续值数目
    num_feature = len(sorted_feature_list)
    # 计算划分点
    dot_feature_list = [round((sorted_feature_list[i][0] + sorted_feature_list[i + 1][0]) / 2.0, 5)
                        for i in range(num_feature - 1)]
    # 计算划分点信息增益
    for dot in dot_feature_list:
        # 根据划分点划分数据集
        low_dataset, high_dataset = split_part_dataset(dataset, feature, dot)

        # 计算两部分的特征值熵和权重的乘积之和
        sun_entropy = len(low_dataset) / len(dataset) * entroy_calc(low_dataset) + \
                      len(high_dataset) / len(dataset) * entroy_calc(high_dataset)

        # 计算出信息增益
        info_gain = father_entropy - sun_entropy

        if info_gain > max_info_gain:
            best_dot = dot
            max_info_gain = info_gain

    return max_info_gain, best_dot


# 选择特征进行划分，对所有特征调用info_gain_calc,得到最好的特征
# 选择特征对应的划分点
def feature_choose(dataset):
    # 得到数据的特征值总数
    num_feature = len(dataset[0]) - 1
    # 计算出父节点信息熵
    father_entropy = entroy_calc(dataset)
    # 基础信息增益为0.0
    best_info_gain = 0.0
    # 最好的特征值
    best_feature = -1
    # 如果是连续值的话，用来记录连续值的划分点
    best_dot = 0.0

    # 对每个特征值进行求信息熵
    for i in range(num_feature):
        info_gain, dot = info_gain_calc(dataset, i, father_entropy)
        if info_gain > best_info_gain:
            # 最优信息增益
            best_info_gain = info_gain
            # 划分的最优特征值
            best_feature = i
            best_dot = dot

    return best_feature, best_dot


def decision_tree(dataset, labels):
    # 拿到所有数据集的分类标签
    class_list = [data[-1] for data in dataset]

    # 统计第一个标签出现的次数并与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 选择最好的划分特征，得到该特征的下标
    best_feature = feature_choose(dataset=dataset)

    # 得到最好特征的名称
    best_feature_label = ''
    # 重新修改分叉点信息
    best_feature_label = str(labels[best_feature[0]]) + '=' + str(best_feature[1])
    # 得到当前的划分点
    dot = best_feature[1]
    # 得到下标值
    best_feature = best_feature[0]
    # 连续值标志

    # 使用一个字典来存储树结构，分叉处为划分的特征名称
    my_tree = {best_feature_label: {}}

    # 将连续值划分为不大于当前划分点和大于当前划分点两部分
    low_dataset, high_dataset = split_part_dataset(dataset, best_feature, dot)
    # 得到剩下的特征标签
    son_labels = labels[:]
    # 递归处理小于划分点的子树
    son_tree = decision_tree(low_dataset, son_labels)
    my_tree[best_feature_label]['<'] = son_tree
    # 递归处理大于当前划分点的子树
    son_tree = decision_tree(high_dataset, son_labels)
    my_tree[best_feature_label]['>'] = son_tree

    return my_tree


if __name__ == '__main__':
    train_dataset, test_dataset, labels = get_dataset(0.3)
    temp_label = labels[:]  # 深拷贝
    Tree = decision_tree(train_dataset, labels)
    treePlotter.create_plot(Tree)
    print(str(Tree))
    f = open('tree.txt', 'w')
    f.write(str(Tree))
    f.close()
    g = open('test_data.txt', 'w')
    g.write(str(test_dataset))
    g.close()
    h = open('temp_label.txt', 'w')
    h.write(str(labels))
    h.close()