import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import *
import random
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
# 定义图像的预处理步骤
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载猫狗样本数据集
sample_data_dir = 'sample_data\\data'  # 数据目录
dataset = datasets.ImageFolder(root=sample_data_dir, transform=transform)

# 确保从每个类提取样本
class_indices = {0: [], 1: []}  # 假设 0 代表猫，1 代表狗
for i, (_, label) in enumerate(dataset):
    class_indices[label].append(i)
# 随机抽取每个类的 5 个样本
train_labeled_indices = []
for label in class_indices:
    sampled_indices = random.sample(class_indices[label], 500)
    train_labeled_indices.extend(sampled_indices)


total_labeled_indices = train_labeled_indices.copy()  # 复制第一次训练的样本索引

# 剩下的样本作为无标签数据
unlabeled_indices = [i for i in range(len(dataset)) if i not in train_labeled_indices]

# 随机抽取 500 个样本作为测试数据集
test_sample_size = 500
test_indices = random.sample(unlabeled_indices, min(test_sample_size, len(unlabeled_indices)))

# 创建测试数据集
test_set = Subset(dataset, test_indices)

# 更新 unlabeled_indices，去掉抽取的样本
unlabeled_indices = [i for i in unlabeled_indices if i not in test_indices]
# 创建有标签和无标签数据集
train_labeled_set = Subset(dataset, train_labeled_indices)
unlabeled_set = Subset(dataset, unlabeled_indices)

#随机从unlabeled_indices中选择N个样本作为观测点
selected_indices_for_visual = random.sample(unlabeled_indices, 10)
selected_entropy_history = []

# 打印一些信息以确认
print(f'测试集样本数量: {len(test_set)}')
print(f'无标签数据集样本数量: {len(unlabeled_set)}')
print(f'测试集样本索引: {test_indices}')
print(f'更新后的无标签数据集样本索引: {unlabeled_indices}')
all_data_number = len(train_labeled_set)+len(unlabeled_set)+len(test_set)
test_number = len(test_set)
# 定义数据加载器
train_labeled_loader = DataLoader(train_labeled_set, batch_size=32, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_set, batch_size=64, shuffle=False)

test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
def calculate_entropy(probs):
    """计算每个样本的熵，输入为模型预测的概率分布"""
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)  # 防止log(0)
    return entropy

# 训练模型函数
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def calculate_accuracy(pred_probs, true_labels):
    pred_labels = np.argmax(pred_probs, axis=1)  # 获取预测标签
    accuracy = np.sum(pred_labels == true_labels) / len(true_labels)  # 计算准确率
    return accuracy
def predict_unlabeled_with_probs(model, dataloader):
    model.eval()
    all_probs = []
    all_labels=[]
    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)  # 转化为概率分布
        all_probs.extend(probs.detach().cpu().numpy())  # 收集所有样本的概率
        all_labels.extend(labels.detach().cpu().numpy())  # 收集真实标签
    accuracy = calculate_accuracy( np.array(all_probs),  np.array(all_labels))
    print(f'Accuracy: {accuracy:.2f}')
    return np.array(all_probs)


def merge_new_labeled_data(total_labeled_indices, pseudo_labels, unlabeled_indices, sample_size=10):
    # 从无标签样本中选择新的样本索引
    available_indices = [i for i in unlabeled_indices if i not in total_labeled_indices]

    if len(available_indices) < sample_size:
        print("没有足够的样本可供选择.")
        return [], []

    new_indices = random.sample(available_indices, sample_size)

    # 获取对应的伪标签
    pseudo_labels_new = [pseudo_labels[i] for i in new_indices]

    return new_indices, pseudo_labels_new


if __name__ == '__main__':
    # 设置超参数

    num_classes =2
    num_epochs = 300  #  总训练轮数    ==N_epochs*(无标签数量/select_sample_num)
    select_sample_num =500  ####每次增加的数量
    N_epochs = 100     ###每轮训练的次数

    # 初始化模型
    model = SimpleCNN()
    model =model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 半监督训练过程

    train_labeled_labels = []
    selected_entropy_history = {}  # 字典来存储每个样本的熵

    # 初始化一个字典来存储每个测试样本的熵值
    test_entropy_history = {index: [] for index in range(test_number)}
    # 迭代数据加载器提取标签
    for images, labels in train_labeled_loader:
        train_labeled_labels.extend(labels.numpy())  # 转换为 NumPy 数组并扩展列表
    average_entropy_history = []  # 用于存储每个 epoch 的平均信息熵
    average_entropy_history_test =[]
    #####训练下初始模型  这里的训练次数用于训练初始模型
    # for epoch in range(10):
    #     loss = train_epoch(model, train_labeled_loader, optimizer, criterion)
    #     print(f'Epoch {epoch + 1}/{N_epochs}, Loss: {loss}')



    for epoch in range(num_epochs):
        # 训练有标签数据
        print(f'当前训练数据数量: {len(train_labeled_set)}')
        loss = train_epoch(model, train_labeled_loader, optimizer, criterion)
        print(f'Epoch {epoch + 1}/{N_epochs}, Loss: {loss}')

        # 对无标签数据进行伪标签预测并计算概率分布
        probs_unlabeled = predict_unlabeled_with_probs(model, unlabeled_loader)

        probs_test = predict_unlabeled_with_probs(model, test_loader)
        entropy_test = calculate_entropy(probs_test)
        # 存储每个测试样本的熵值
        for index in range(len(entropy_test)):
            test_entropy_history[index].append(entropy_test[index])

        avg_entropy_test = np.mean(entropy_test)
        print(f'平均test信息熵（无接触数据）: {avg_entropy_test}')
        average_entropy_history_test.append(avg_entropy_test)
        # 计算无标签样本的信息熵
        entropy_unlabeled = calculate_entropy(probs_unlabeled)
        avg_entropy = np.mean(entropy_unlabeled)
        print(f'平均信息熵（无标签数据）: {avg_entropy}')
        average_entropy_history.append(avg_entropy)

        # 获取对应样本的概率
        probs_unlabeled_all = np.full((all_data_number, num_classes), -1, dtype=np.float32)  # 初始化为-1，表示未知标签
 #  # 初始化为-1，表示未知标签
        for i, unlabeled_index in enumerate(unlabeled_indices):
            probs_unlabeled_all[unlabeled_index] = probs_unlabeled[i]  # 直接使用无标签样本的概率

        selected_probs = probs_unlabeled_all[selected_indices_for_visual]
        selected_entropy = calculate_entropy(selected_probs)
        # 存储每个样本的熵
        for index, entropy in zip(selected_indices_for_visual, selected_entropy):
            if index not in selected_entropy_history:
                selected_entropy_history[index] = []  # 初始化列表
            selected_entropy_history[index].append(entropy)  # 添加当前epoch的熵


        # 每N个epoch后，从无标签数据中选取样本加入训练集
        if (epoch + 1) % N_epochs == 0 :

            print('Updating training set with pseudo-labeled data...')
            pseudo_labels = np.full(all_data_number, -1)  # 初始化为-1，表示未知标签

            # 将前10个样本的真实标签填入伪标签数组
            for i, index in enumerate(train_labeled_indices):
                pseudo_labels[index] = train_labeled_labels[i]  # 使用原始索引放置真实标签
            for i, unlabeled_index in enumerate(unlabeled_indices):
                pseudo_labels[unlabeled_index] = np.argmax(probs_unlabeled[i])
                # 新增的样本索引
            new_indices, pseudo_labels_new = merge_new_labeled_data(total_labeled_indices, pseudo_labels,
                                                                    unlabeled_indices, sample_size=select_sample_num)

            # 合并已有的样本索引与新增样本索引
            total_labeled_indices = list(set(total_labeled_indices) | set(new_indices))

            # 更新训练集
            train_labeled_set = Subset(dataset, total_labeled_indices)
            train_labeled_loader = DataLoader(train_labeled_set, batch_size=32, shuffle=True)

    print('训练结束')

    # 创建一个包含四个子图的图形
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    # 第一个子图：平均信息熵
    axs[0].plot(range(num_epochs), average_entropy_history, marker='o', color='blue', label='Average Entropy')
    axs[0].set_title('Average Information Entropy Over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Average Information Entropy')
    axs[0].set_xticks(range(num_epochs))
    axs[0].grid()

    # 每10个epoch标记一个点
    for i in range(9, num_epochs, 10):
        axs[0].scatter(i, average_entropy_history_test[i], color='red', label='Update Point' if i == 9 else "",
                       zorder=5)

    axs[0].legend()

    # 第二个子图：选定无标签样本的信息熵
    sample_indices = list(selected_entropy_history.keys())[:3]

    for sample_index in sample_indices:
        entropy_values = selected_entropy_history[sample_index]
        axs[1].plot(range(len(entropy_values)), entropy_values, marker='o', label=f'Sample {sample_index}')
        axs[1].scatter(range(min(3, len(entropy_values))), entropy_values[:3], color='red', zorder=5)

    axs[1].set_title('Entropy of Selected Unlabeled Samples Over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Entropy')
    axs[1].set_xticks(range(num_epochs))
    axs[1].grid()
    axs[1].legend()

    # 第三个子图：测试数据的平均信息熵
    axs[2].plot(range(num_epochs), average_entropy_history_test, marker='o', color='blue', label='Average Test Entropy')
    axs[2].set_title('Average Information Entropy for Test Data Over Epochs')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Average Information Entropy')
    axs[2].set_xticks(range(num_epochs))
    axs[2].grid()

    # 每10个epoch标记一个点
    for i in range(9, num_epochs, 10):
        axs[2].scatter(i, average_entropy_history_test[i], color='red', label='Update Point' if i == 9 else "",
                       zorder=5)

    axs[2].legend()

    # 第四个子图：测试样本的信息熵
    test_sample_indices = list(test_entropy_history.keys())[:3]

    for sample_index in test_sample_indices:
        entropy_values = test_entropy_history[sample_index]
        axs[3].plot(range(len(entropy_values)), entropy_values, marker='o', label=f'Test Sample {sample_index}')
        axs[3].scatter(range(min(3, len(entropy_values))), entropy_values[:3], color='red', zorder=5)

    axs[3].set_title('Entropy of Selected Test Samples Over Epochs')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Entropy')
    axs[3].set_xticks(range(num_epochs))
    axs[3].grid()
    axs[3].legend()

    plt.tight_layout()  # 自动调整子图参数，使其填充整个图像区域
    plt.show()
