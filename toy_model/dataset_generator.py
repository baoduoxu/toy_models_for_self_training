import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

class DatasetGenerator:
    def __init__(self, mu, n_labelled, n_unlabelled):
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.dim = len(mu)
        self.n_labelled = n_labelled
        self.n_unlabelled = n_unlabelled

    def generate_labelled(self):
        labels = torch.randint(0, 2, (self.n_labelled,)) * 2 - 1  # y ~ Uniform({-1, 1})
        # data = labels.unsqueeze(1) * self.mu + torch.randn(self.n_labelled, 2)  # x | y ~ N(y * mu, I)
        # 生成 d 维数据
        data = labels.unsqueeze(1) * self.mu + torch.randn(self.n_labelled, self.dim)
        return data, labels

    def generate_unlabelled(self):
        labels = torch.randint(0, 2, (self.n_unlabelled,)) * 2 - 1
        # data = labels.unsqueeze(1) * self.mu + torch.randn(self.n_unlabelled, 2)
        data = labels.unsqueeze(1) * self.mu + torch.randn(self.n_unlabelled, self.dim)
        return data
    
    def generate_test_data_near_bayes_optimal_classifier(self, n_test=2, eps=0.5):
        # 逐个生成 n_test 个 服从 y ~ Uniform({-1, 1}), x | y ~ N(y * mu, I) 的测试样本
        cnt = 0
        labels = []
        data = []
        while True:
            tmp_label = torch.randint(0, 2, (1,)) * 2 - 1
            tmp_data = tmp_label * self.mu + torch.randn(1, self.dim)
            if torch.abs(tmp_data[:,0]+tmp_data[:,1]) < torch.tensor(eps):
                cnt += 1
                labels.append(tmp_label)
                data.append(tmp_data)
            if cnt == n_test:
                break
        # 保存到当前文件夹下的 test_data.txt 文件中
        with open('test_data.txt', 'w') as f:
            for i in range(n_test):
                f.write(f'{data[i][0][0].item()} {data[i][0][1].item()} {labels[i].item()}\n')
        return data, labels
    
    def plot_data(self, labelled_data, labelled_labels, unlabelled_data):
        # 将有标签数据分开，分别绘制红色和蓝色
        plt.scatter(unlabelled_data[:, 0], unlabelled_data[:, 1], color='gray', alpha=0.5, label='Unlabelled')
        plt.scatter(labelled_data[labelled_labels == 1, 0], labelled_data[labelled_labels == 1, 1], color='red', label='Label 1')
        plt.scatter(labelled_data[labelled_labels == -1, 0], labelled_data[labelled_labels == -1, 1], color='blue', label='Label -1')
        plt.legend()
        plt.title('Scatter Plot of Labelled and Unlabelled Data')
        plt.show()