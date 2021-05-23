

# 输入三个变量（决策树，属性特征标签，测试的数据）
# 递归计算将数据带入树中，求分类输出
def test_classify(input_tree, feature_labels, test_data):
    # 获取决策树第一层划分依据
    feature_dot = list(input_tree.keys())[0]
    # 获取特征子树
    dis_tree = input_tree[feature_dot]
    # 获取决策树第一层划分特征的index
    feature_index = feature_labels.index(feature_dot[:feature_dot.index('=')])

    for key in dis_tree.keys():
        if test_data[feature_index] > float(feature_dot[feature_dot.index('=') + 1:]):
            # 判断是否搜索结束
            if type(dis_tree['>']).__name__ == 'dict':
                final_label = test_classify(dis_tree['>'], feature_labels, test_data)
            else:
                final_label = dis_tree['>']
            return final_label
        else:
            if type(dis_tree['<']).__name__ == 'dict':
                final_label = test_classify(dis_tree['<'], feature_labels, test_data)
            else:
                final_label = dis_tree['<']
            return final_label


def confusion_matrix(my_tree, label, test_set):
    # 二维混淆矩阵
    con_mat = {'Iris-setosa': {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0},
               'Iris-versicolor': {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0},
               'Iris-virginica': {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}}
    for data in test_set:
        predict = test_classify(my_tree, label, data)  # predict为预测结果
        actual = data[-1]  # actual为实际结果
        con_mat[actual][predict] += 1  # 计算confusion matrix
    return con_mat


# 准确度
def precision(classes, matrix):
    precision_dict = {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}
    all_predict_num = 0  # 例如存储当预测值为Iris-setosa时的所有情况数量
    for item in classes:
        true_predict_num = matrix[item][item]  # 准确预测数量
        all_predict_num = 0
        for temp_item in classes:
            all_predict_num += matrix[temp_item][item]
        precision_dict[item] = round(true_predict_num / all_predict_num, 2)
    return precision_dict


# 召回率
def recall(classes, matrix):
    recall_dict = {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}
    for item in classes:
        true_predict_num = matrix[item][item]  # 准确预测数量
        all_predict_num = 0  # 例如存储当预测值为Iris-setosa时的所有情况数量
        for temp_item in classes:
            all_predict_num += matrix[item][temp_item]
        recall_dict[item] = round(true_predict_num / all_predict_num, 2)
    return recall_dict


def f1_score(p, f):
    return round(2*p*f / (p+f), 2)


def accuracy(classes, matrix):
    right_num = 0
    sum_num = 0
    for row in classes:
        right_num += matrix[row][row]
        for column in classes:
            sum_num += matrix[row][column]
    return round(right_num / sum_num, 2)


# 展示结果
def show_result(precision, recall, classes):
    print('\t\t\t\t', 'precision', '\t', 'recall', '\t', 'F1_score')
    for item in classes:
        print(item, '\t', precision[item], '\t', recall[item], '\t', f1_score(precision[item], recall[item]))


if __name__ == "__main__":
    f = open('tree.txt', 'r')
    Tree = eval(f.read())
    f.close()
    g = open('test_data.txt', 'r')
    test_dataset = eval(g.read())
    g.close()
    h = open('temp_label.txt', 'r')
    temp_label = eval(h.read())
    h.close()
    matrix = confusion_matrix(Tree, temp_label, test_dataset)  # confusion matrix
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']  # 所有分类
    precision_value = precision(classes, matrix)
    recall_value = recall(classes, matrix)
    show_result(precision_value, recall_value, classes)
    print('aaccuracy = ' + str(accuracy(classes, matrix)))