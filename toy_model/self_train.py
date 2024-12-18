import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt


class SelfTraining:
    def __init__(self, eta, sigma, B, T, test_data, test_labels):
        self.eta = eta
        self.sigma = sigma
        self.B = B
        self.T = T
        # test_data 是一个 list, 每个元素是一个 tensor, 表示若干个测试样本, 现在将其转换为一个 tensor
        # print(f'test_data: {len(test_data[0])}')
        # self.test_data = torch.stack(test_data[0], dim=0)
        self.test_data = test_data
        self.test_labels = test_labels
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
        initial_classifier = torch.tensor([1.1, -0.5], dtype=torch.float32)
        # initial_classifier = [1.0, 0.25]
        beta = initial_classifier / torch.norm(initial_classifier)  # 归一化
        print(f'initial classifier after normalization: {beta}')
        # 创建 entrpoy_list 是一个形状为 (n_test, T) 的张量, 每一列表示每个测试样本在每次迭代中的熵
        entropy_list = torch.zeros((self.test_data.size(0), self.T))
        
        unlabelled_batches = torch.split(unlabelled_data, self.B)  # 将数据划分为 batch

        for t in range(self.T):
            x_batch = unlabelled_batches[t]  # 取当前 batch
            pseudo_labels = torch.sign(torch.matmul(x_batch, beta))  # 生成伪标签
            # TODO: 计算伪标签的正确率
            grad_sum = torch.zeros_like(beta)
            for i in range(self.B):
                pred = torch.matmul(x_batch[i], beta)
                grad_sum += (1 / self.sigma) * pseudo_labels[i] * x_batch[i] * (-torch.sigmoid(-(1 / self.sigma) * pseudo_labels[i] * pred))

            beta -= self.eta / self.B * grad_sum
            # print(f'Batch/iteration {t}, beta before normalize: {beta}')
            beta = beta / torch.norm(beta)  # 权重归一化
            print(f'Batch/iteration {t}, beta after normalize: {beta}')

        # 计算测试数据的概率并保存到熵列表
            prob = torch.sigmoid(torch.matmul(self.test_data, beta))
            entropy = -prob * torch.log(prob) - (1 - prob) * torch.log(1 - prob)
            # print(f'prob shape: {prob.shape}')
            # print(f'prob: {prob}')
            # entropy_list[:, t] = prob.squeeze()
            entropy_list[:, t] = entropy.squeeze()

        return beta, entropy_list


    # def plot_entropy(self, entropy_list, test_data):
    #     '''
    #     画出所有测试样本迭代过程中的熵的变化, 其中 entropy_list 是一个形状为  (n_test, T) 的张量,
    #     表示 n_test 个样本在 T 轮迭代过程中熵的变化. 请画出折线图, 表示这 n_test 个样本的熵的变化曲线,
    #     同时计算它们熵的每一轮迭代的平均值, 也同时画出这个均值的在每一轮迭代过程中变化的折线图.
    #     '''
    #     n_test, T = entropy_list.shape  # 获取样本数和迭代次数
    #     # avg_entropy = torch.mean(entropy_list, dim=0).detach().numpy()  # 计算每轮迭代的熵均值
    #     print(f'enropy_list shape: {entropy_list.shape}')
    #     # 画每个样本的熵变化折线图
    #     plt.figure(figsize=(10, 6))
    #     print(f'n_test: {n_test}')
    #     print(f'enropy_list: {entropy_list}')
    #     for i in range(n_test):
    #         plt.plot(range(T), entropy_list[i].detach().numpy(), label=f'Sample {i+1}: {self.test_data.numpy()[i]} with label {int(self.test_labels[i])}', alpha=0.3)

    #     # 画熵均值变化的折线图
    #     # plt.plot(range(T), avg_entropy, color='black', label='Average Entropy', linewidth=2)

    #     # 图形标题和标签
    #     plt.title('Entropy Change Over Iterations for Test Samples', fontsize=14)
    #     plt.xlabel('Iteration', fontsize=12)
    #     plt.ylabel('Entropy', fontsize=12)
    #     plt.legend(loc='upper right', fontsize=10)
    #     plt.grid(True)
    #     plt.show()

    def plot_entropy(self, entropy_list, test_data):
        '''
        画出所有测试样本迭代过程中的熵的变化, 其中 entropy_list 是一个形状为  (n_test, T) 的张量,
        表示 n_test 个样本在 T 轮迭代过程中熵的变化. 请画出折线图, 表示这 n_test 个样本的熵的变化曲线,
        同时计算它们熵的每一轮迭代的平均值, 也同时画出这个均值的在每一轮迭代过程中变化的折线图.
        '''
        # 设置字体为 Times New Roman
        matplotlib.rc('font', family='Times New Roman')
        
        n_test, T = entropy_list.shape  # 获取样本数和迭代次数
        avg_entropy = torch.mean(entropy_list, dim=0).detach().numpy()  # 计算每轮迭代的熵均值

        # 创建画布
        plt.figure(figsize=(9, 6))
        
        # 每个样本的熵变化折线图
        for i in range(n_test):
            plt.plot(
                range(T), 
                entropy_list[i].detach().numpy(), 
                label=f'Sample {i+1}: {test_data.numpy()[i]} with label {int(self.test_labels[i])}', 
                alpha=0.8,  # 减小透明度以增强颜色
                linewidth=3  # 加粗线条
            )
        
        # # 画熵均值变化的折线图
        # plt.plot(
        #     range(T), 
        #     avg_entropy, 
        #     color='black', 
        #     label='Average Entropy', 
        #     linewidth=4,  # 加粗线条
        #     linestyle='--'  # 虚线表示均值
        # )

        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        
        # 图形标题和轴标签
        # plt.title('Entropy Change Over Iterations for Test Samples', fontsize=16)
        plt.xlabel('Number of Iterations', fontsize=24)
        plt.ylabel('Entropy', fontsize=24)

        # 设置图例，放置在图的右上角偏下
        plt.legend(
            loc='upper right', 
            bbox_to_anchor=(1.0, 0.9),  # 将图例稍微向下移动
            fontsize=18, 
            frameon=True  # 去掉图例框线
        )
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 显示图形
        plt.tight_layout()
        plt.show()

    # def plot_entropy(self, entropy_list, test_data):
        # '''
        # 画出所有测试样本迭代过程中的熵的变化, 其中 entropy_list 是一个形状为  (n_test, T) 的张量,
        # 表示 n_test 个样本在 T 轮迭代过程中熵的变化. 请画出折线图, 表示这 n_test 个样本的熵的变化曲线,
        # 同时计算它们熵的每一轮迭代的平均值, 也同时画出这个均值的在每一轮迭代过程中变化的折线图.
        # '''
        # # 设置字体为 Times New Roman
        # matplotlib.rc('font', family='Times New Roman')
        
        # n_test, T = entropy_list.shape  # 获取样本数和迭代次数
        # avg_entropy = torch.mean(entropy_list, dim=0).detach().numpy()  # 计算每轮迭代的熵均值

        # # 创建画布，设置更紧凑的尺寸
        # plt.figure(figsize=(14, 8))  # 稍微增大画布的宽高

        # # 每个样本的熵变化折线图
        # for i in range(n_test):
        #     plt.plot(
        #         range(T), 
        #         entropy_list[i].detach().numpy(), 
        #         label=f'Sample {i+1}: {test_data.numpy()[i]} with label {int(self.test_labels[i])}', 
        #         alpha=0.8,  # 增强颜色的可见性
        #         linewidth=2  # 加粗线条
        #     )
        
        # # 画熵均值变化的折线图
        # plt.plot(
        #     range(T), 
        #     avg_entropy, 
        #     color='black', 
        #     label='Average Entropy', 
        #     linewidth=3,  # 加粗均值曲线
        #     linestyle='--'  # 虚线表示均值
        # )
        
        # # 图形标题和轴标签
        # # plt.title('Entropy Change Over Iterations for Test Samples', fontsize=42)
        # plt.xlabel('Number of Iterations', fontsize=20)
        # plt.ylabel('Entropy', fontsize=20)

        # # 设置横纵坐标数字的字体大小
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)

        # # 设置图例，放置在图的右上角偏下
        # plt.legend(
        #     loc='upper right', 
        #     bbox_to_anchor=(1.0, 0.8),  # 将图例稍微向下移动
        #     fontsize=16,  # 增大图例字体
        #     frameon=False  # 去掉图例框线
        # )
        
        # # 添加网格，设置网格线的样式和透明度
        # plt.grid(True, linestyle='--', alpha=0.7)

        # # 使用 tight_layout 来使得图形更加紧凑
        # plt.tight_layout()
        # plt.show()