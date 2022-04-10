import numpy as np
import torch

"""
使用numpy模拟整个过程:

"""

def compute_error_for_line_given_points(b, w, points):
    """
    用来计算平均损失的
    :param b: 偏置
    :param w: 权重
    :param points: 样本点[[x1, y1],[],...[xi, yi]]
    :return: 返回样本总平均损失.
    """

    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))  # 返回平均损失


# 计算梯度, 更新模型参数, b和w
def step_gradient(b_current, w_current, points, learning_rate):
    """
    梯度下降
    :param b_current: 当前的b
    :param w_current: 当前模型参数的w
    :param points: 样本点, x,y
    :param learning_rate: 学习率, 步进长度
    :return: 更新后的b和w
    """
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # loss: (WX + b - Y) ^ 2
        # 对w求导: 2*(WX + b -Y) * X
        # 对b求导: 2*(WX + b -Y)
        # 然后都除以N, 表示平均值. 累加上去也得除以N, 所以就在这除以N了
        b_gradient += -(2 / N) * (y - (w_current * x) + b_current)
        w_gradient += -(2 / N) * x * (y - (w_current * x) + b_current)
    # 更新后的b和w
    new_b = b_current - (learning_rate * b_gradient)
    new_w = w_current - (learning_rate * w_gradient)

    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    """
    使用梯度下降算法, 迭代b和w
    :param points: 样本点
    :param starting_b: 初始的b
    :param starting_w: 初始值 w
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数,
    :return: 返回迭代好的b和w
    """
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, m = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    points = np.genfromtxt("", delimiter=',')  # 读取数据
    learning_rate = 0.0001  # 学习率
    # 设置初始值
    initial_b = 0
    initial_w = 0
    num_iterations = 1000

    print("初始化: running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print(f"运行结果: b:{b} w:{w}")


if __name__ == '__main__':
    run()
