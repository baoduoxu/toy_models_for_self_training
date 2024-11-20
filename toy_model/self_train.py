import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class SelfTraining:
    def __init__(self, eta, sigma, B, T, test_data):
        self.eta = eta
        self.sigma = sigma
        self.B = B
        self.T = T
        # test_data 是一个 list, 每个元素是一个 tensor, 表示若干个测试样本, 现在将其转换为一个 tensor
        # print(f'test_data: {len(test_data[0])}')
        self.test_data = torch.stack(test_data[0], dim=0)
        # self.test_data = test_data[0]
    def plot_pseudolabelled_data(self, x_batch, pseudo_labels, t):
        # 将伪标签为 1 的样本标为淡红色, 伪标签为 -1 的样本标为淡蓝色
        plt.scatter(x_batch[pseudo_labels == 1, 0], x_batch[pseudo_labels == 1, 1], color='lightcoral', alpha=0.5)
        plt.scatter(x_batch[pseudo_labels == -1, 0], x_batch[pseudo_labels == -1, 1], color='lightblue', alpha=0.5)
        plt.title(f'Pseudo-labelled Data at Iteration {t+1}')
        plt.legend()
        plt.show(block=False)  # 非阻塞方式展示图像
        plt.pause(0.1)  # 稍作暂停以显示图像

    def train(self, unlabelled_data, initial_classifier):
        print(f'Initial classifier: {initial_classifier}')
        # initial_classifier = [0.1, 0.9]
        beta = initial_classifier / torch.norm(initial_classifier)  # 归一化
        print(f'initial classifier after normalization: {beta}')
        # entropy_list = [[] for _ in range(len(self.test_data))]
        # 创建 entrpoy_list 是一个形状为 (n_test, T) 的张量, 每一列表示每个测试样本在每次迭代中的熵
        entropy_list = torch.zeros((self.test_data.size(0), self.T))
        for t in range(self.T):
            idx = torch.randint(0, unlabelled_data.size(0), (self.B,))
            x_batch = unlabelled_data[idx]
            pseudo_labels = torch.sign(torch.matmul(x_batch, beta))  # 伪标签
            
            # self.plot_pseudolabelled_data(x_batch, pseudo_labels, t)
            
            grad_sum = torch.zeros_like(beta)
            for i in range(self.B):
                pred = torch.matmul(x_batch[i], beta)
                grad_sum += pseudo_labels[i] * x_batch[i] * (1 / self.sigma) * torch.sigmoid(-pseudo_labels[i] * pred)
                
            beta -= self.eta / self.B * grad_sum
            beta = beta / torch.norm(beta)  # 权重归一化
            print(f'Iteration {t}, beta: {beta}')
            # print(f'test data: {self.test_data[0]}')
            prob = torch.sigmoid(torch.matmul(self.test_data, beta))
            # print(f'Probability of test data being in class 1: {prob}')
            entrpoy = -prob * torch.log(prob) - (1 - prob) * torch.log(1 - prob)
            # entropy 应该是一个形状为 (n_test,) 的 tensor, 每个元素表示每个样本的熵, 需要将 entropy 赋值给 entropy_list 的第 t 列
            # print(f'Entropy: {entrpoy.shape}')
            # entropy_list[:, t] = prob.squeeze()
            entropy_list[:, t] = entrpoy.squeeze()
            # print(f'Entropy: {entrpoy}')
            # print(f'Probability of test data being in class 1: {prob}; its true label: {self.test_data[1][0]}')
        return beta, entropy_list

    def plot_entropy(self, entropy_list, test_data):
        '''
        画出所有测试样本迭代过程中的熵的变化, 其中 entropy_list 是一个形状为  (n_test, T) 的张量,
        表示 n_test 个样本在 T 轮迭代过程中熵的变化. 请画出折线图, 表示这 n_test 个样本的熵的变化曲线,
        同时计算它们熵的每一轮迭代的平均值, 也同时画出这个均值的在每一轮迭代过程中变化的折线图.
        '''

        n_test, T = entropy_list.shape  # 获取样本数和迭代次数
        avg_entropy = torch.mean(entropy_list, dim=0).detach().numpy()  # 计算每轮迭代的熵均值

        # 画每个样本的熵变化折线图
        plt.figure(figsize=(10, 6))
        for i in range(n_test):
            # print(test_data[0][i])
            plt.plot(range(T), entropy_list[i].detach().numpy(), label=f'Sample {i+1} with label {int(test_data[1][i])}', alpha=0.3)

        # 画熵均值变化的折线图
        plt.plot(range(T), avg_entropy, color='black', label='Average Entropy', linewidth=2)

        # 图形标题和标签
        plt.title('Entropy Change Over Iterations for Test Samples', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Entropy', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True)
        plt.show()

