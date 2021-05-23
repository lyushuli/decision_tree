import matplotlib.pyplot as plt

# 决策节点样式
decisionNode = dict(boxstyle="darrow", fc="0.8")
# 结果节点样式
leafNode = dict(boxstyle="sawtooth", fc="0.7")
# 箭头样式
arrow_args = dict(arrowstyle="<|-")


# 获取树的叶子节点
def get_leafs_num(tree):
    num_leafs = 0
    first_dot = list(tree.keys())
    first_feature_dot = first_dot[0]
    sub_tree = tree[first_feature_dot]
    for key in sub_tree.keys():
        # 判断是否是叶子节点（通过类型判断，子类不存在，则类型为str；子类存在，则为dict）
        if type(sub_tree[key]).__name__ == 'dict':
            num_leafs += get_leafs_num(sub_tree[key])
        else:
            num_leafs += 1
    return num_leafs


# 获取树的层数
def get_layer_num(tree):
    layer_num = 0
    # dict转化为list
    first_dot = list(tree.keys())
    first_feature_dot = first_dot[0]
    sub_tree = tree[first_feature_dot]
    for key in sub_tree.keys():
        if type(sub_tree[key]).__name__ == 'dict':
            current_num = 1 + get_layer_num(sub_tree[key])
        else:
            current_num = 1
        if current_num > layer_num:
            layer_num = current_num
    return layer_num


# 绘制节点
# nodetext为要显示的文本，center_point当前点
def plot_node(node_text, center_point, parent_point, nodeType):
    create_plot.ax1.annotate(node_text, xy=parent_point, xycoords='axes fraction',
                            xytext=center_point, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 文本绘制
def plot_text(center_point, parent_point, txt):
    x = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
    y = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
    create_plot.ax1.text(x, y, txt, va="center", ha="center", rotation=30)


def plot_tree(tree, parent_point, node_text):
    num_leafs = get_leafs_num(tree)
    depth = get_layer_num(tree)
    first_dot = list(tree.keys())
    first_feature_dot = first_dot[0]
    # 确定中心点
    center_point = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    # 绘制文本
    plot_text(center_point, parent_point, node_text)
    # 绘制决策节点
    plot_node(first_feature_dot, center_point, parent_point, decisionNode)
    sub_tree = tree[first_feature_dot]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in sub_tree.keys():
        # 递归搜树
        # 终止条件
        if type(sub_tree[key]).__name__ == 'dict':
            plot_tree(sub_tree[key], center_point, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            # 绘制分类节点
            plot_node(sub_tree[key], (plot_tree.xOff, plot_tree.yOff), center_point, leafNode)
            plot_text((plot_tree.xOff, plot_tree.yOff), center_point, str(key))
    plot_tree.yOff = plot_tree.yOff + 0.5 / plot_tree.totalD


# 绘制决策树
def create_plot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_leafs_num(tree))
    plot_tree.totalD = float(get_layer_num(tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()
