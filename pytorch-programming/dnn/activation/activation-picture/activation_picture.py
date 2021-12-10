import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.where(x < 0, 0, x)


def plot_relu():
    x = np.arange(-10, 10, 0.1)
    y = relu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.plot(x, y, color="black")
    plt.xlim([-10.05, 10.05])
    plt.ylim([0, 10.02])
    ax.set_yticks([2, 4, 6, 8, 10])
    plt.tight_layout()
    plt.legend(['Relu'])
    plt.savefig("relu.png")
    plt.show()


def plot_sigmoid():
    x = np.linspace(-10, 10, 1000)  # 这个表示在-10到10之间生成1000个x值
    y = sigmoid(x)  # 对上述生成的1000个数循环用sigmoid公式求对应的y
    plt.xlim((-10, 10))
    plt.ylim((0.00, 1.00))
    plt.yticks([0.5, 1.0], [0.5, 1.0])  # 设置y轴显示的刻度
    plt.plot(x, y, color="black")  # 用上述生成的1000个xy值对生成1000个点
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # 删除右边框设为无
    ax.spines['top'].set_color('none')  # 删除上边框设为无
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))  # 调整x轴位置
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))  # 调整y轴位置
    plt.legend(['Sigmoid'])
    plt.savefig("sigmoid.png")
    plt.show()


def plot_tanh():
    x = np.arange(-10, 10, 0.1)
    y = tanh(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color="black")
    plt.xlim([-10.05, 10.05])
    plt.ylim([-1.02, 1.02])
    ax.set_yticks([-1.0, -0.5, 0.5, 1.0])
    ax.set_xticks([-10, -5, 5, 10])
    plt.tight_layout()
    plt.legend(['Tanh'])
    plt.savefig("tanh.png")
    plt.show()


if __name__ == "__main__":
    plot_relu()
    plot_sigmoid()
    plot_tanh()
