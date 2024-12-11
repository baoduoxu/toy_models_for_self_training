import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset_generator import DatasetGenerator
from get_init_classifier import LogisticRegressionSGD
from self_train import SelfTraining

def is_mostly_decreasing(sequence, threshold=0.95):
    """
    判断序列是否呈现整体递减趋势。通过判断相邻元素递减的比例是否超过 threshold。
    """
    n = len(sequence)
    if n < 2:
        return True  # 序列过短可以认为递减
    count_decreasing = sum(1 for i in range(1, n) if sequence[i-1] > sequence[i])
    ratio = count_decreasing / (n - 1)
    return ratio >= threshold

def analyze_entropy_trend(entropy_list, threshold=0.9):
    """
    分析每个样本的熵变化趋势，判断是否整体递减。
    同时计算每一轮熵的平均值，并判断平均值是否呈递减趋势。
    
    参数：
    entropy_list: 形状为 (n_test, T) 的张量，表示 n_test 个样本在 T 轮迭代中的熵变化。
    threshold: 判断整体递减的阈值，表示需要多少比例的递减对才能判断为递减。

    返回：
    sample_trends: 一个布尔值列表，表示每个样本的熵是否递减。
    avg_trend: 布尔值，表示熵的平均值是否呈现递减趋势。
    """
    n_test, T = entropy_list.shape
    
    # 计算每个样本的熵变化趋势
    sample_trends = [is_mostly_decreasing(entropy_list[i].detach().numpy(), threshold) for i in range(n_test)]
    
    # 计算每轮熵的平均值
    avg_entropy = torch.mean(entropy_list, dim=0).detach().numpy()
    
    # 判断平均值是否递减
    avg_trend = is_mostly_decreasing(avg_entropy, threshold)
    
    return sample_trends, avg_trend

# 参数设置

# mu = [2,1]
mu = [1, 3.023529]
dim = len(mu)
n_labelled = 2*dim # O(d)
epsilon=0.005
n_unlabelled = int(dim*epsilon**(-2)) # O(d*eps^{-2}) 0.01
delta = 0.9
eta = 0.05
B = int(epsilon**(-1))
T = n_unlabelled // B   # iteration num: 400
sigma = 1.0

# 生成数据集
dataset_gen = DatasetGenerator(mu, n_labelled, n_unlabelled)
labelled_data, labelled_labels = dataset_gen.generate_labelled()
unlabelled_data = dataset_gen.generate_unlabelled()


# dataset_gen.plot_data(labelled_data, labelled_labels, unlabelled_data)

# 使用带标签数据训练初始分类器
log_reg_sgd = LogisticRegressionSGD(input_dim=dim, delta=delta, eta=eta, T=T)
initial_classifiers = log_reg_sgd.train(labelled_data, labelled_labels)

# 生成 test_data
# test_data = dataset_gen.generate_labelled()
# # 生成贝叶斯最优分类器附近的 test_data
# test_data = dataset_gen.generate_test_data_near_bayes_optimal_classifier(n_test=20, eps=0.1)
# 按照 label 给 test_data 排序

# # 四个测试数据:
# # [0.5,1.5] 标签为 1; [2, 0] 标签为 1; [-0.5, -1.5] 标签为 -1; [-2, 0] 标签为 -1
# test_data = torch.tensor([[1.5, 1.5], [2, 0], [0.5, -1.5], [-1.5, 0.3]], dtype=torch.float32)
# test_labels = torch.tensor([1, 1, -1, -1], dtype=torch.float32)


# # 生成测试数据:
# # 直线方程
# def line_1(x):
#     return (-9104 / 0.4138) * x
# def line_2(x):
#     return 2 * x
# # 采样10个点
# n_samples = 10
# x_vals = np.random.uniform(-10, 10, n_samples)  # 在区间[-10, 10]内均匀随机采样10个x值
# # 随机选择每个x值对应的y值, 在两条直线之间取值
# y_vals = np.zeros(n_samples)
# for i in range(n_samples):
#     y1 = line_1(x_vals[i])
#     y2 = line_2(x_vals[i])
#     # 在y1和y2之间随机生成一个y值
#     y_vals[i] = np.random.uniform(min(y1, y2), max(y1, y2))
# # 将x和y值组合成二维数组
# points = np.vstack((x_vals, y_vals)).T
# # 转换为torch tensor
# test_data = torch.tensor(points, dtype=torch.float32)
# print("---------------------------------------------")
# print(test_data)


test_sample_num = 4

# test_dataset_gen = DatasetGenerator(mu, test_sample_num, n_unlabelled)

# test_data, test_labels = test_dataset_gen.generate_labelled()

test_data = torch.tensor([[2.2, -6.1],
        [3.2, 5.9],
        [-6.9, -1.7],
        [6.5, -3.5]])

# test_data = torch.tensor([[1, 3.023529], # 一二改成 beta_init 和 mu, 看不在区域中央的熵的变化情况
#         [-5.38, 6.37],
#         [-6.9, -1.7],
#         [6.5, -3.5]])

test_labels = torch.tensor([ -1, 1, -1, 1])
regions = [1, 2, 3, 4]


print(test_data)

print(test_labels)

# exit(0)


# def line_1(x):
#     return -0.84458398 * x # beta_init
# def line_2(x):
#     return 3.02352941 * x # mu
# # 判断每个点所在的区域
# regions = []
# for point in test_data:
#     x, y = point
#     # 判断与直线的位置关系
#     above_line_1 = (y > line_1(x))
#     above_line_2 = (y > line_2(x))

#     # 区域判断
#     if above_line_1==1 and above_line_2==1:
#         regions.append(1)  # 区域1
#     elif above_line_1==1 and above_line_2==0:
#         regions.append(2)  # 区域2
#     elif above_line_1==0 and above_line_2==0:
#         regions.append(3)  # 区域3
#     elif above_line_1==0 and above_line_2==1:
#         regions.append(4)  # 区域4
#     else:
#         exit(0)

# 输出结果
print("区域分布：", regions)

print("test_data", test_data.shape)

print("test_labels", test_labels)


data_for_avg_sample_num = 1000

dataset_for_avg = DatasetGenerator(mu, data_for_avg_sample_num, n_unlabelled)

data_for_avg, labels_for_avg = dataset_for_avg.generate_labelled()

# print("---------------------------------------------")

# print(data_for_avg.shape)
# print(labels_for_avg.shape)
# exit(0)




# 使用无标签数据进行自训练
print(f'eta: {eta}, sigma: {sigma}, B: {B}, T: {T}')
self_trainer = SelfTraining(eta=eta, sigma=sigma, B=B, T=T, test_data=test_data, test_labels=test_labels, data_for_avg=data_for_avg, labels_for_avg=labels_for_avg)
final_classifier, entropy_list, pred_label_list = self_trainer.train(unlabelled_data, initial_classifiers[0])

# 绘制四类样本的折线图
self_trainer.plot_entropy_test(entropy_list, pred_label_list, test_data, regions, test_labels, data_for_avg, labels_for_avg)
# print(analyze_entropy_trend(entropy_list))
print(f'test data: {test_data}')

# 用 final_classifier 对 test_data 进行分类, 并与真实值对比
pred = torch.sign(torch.matmul(torch.Tensor(test_data), final_classifier))
print(f'Predictions: {pred}')
print(f'True labels: {test_labels}')
# 计算正确率
accuracy = torch.mean((pred == test_labels).float())
print(f'Accuracy: {accuracy}')

print(final_classifier)
print(f'ininital classifier: {initial_classifiers[0]/torch.norm(initial_classifiers[0])}')
print(f'Bayes optimal classifier: {dataset_gen.mu / torch.norm(dataset_gen.mu)}, distance between the final classifier and the Bayes optimal classifier: {torch.norm(final_classifier - dataset_gen.mu/torch.norm(dataset_gen.mu))}')