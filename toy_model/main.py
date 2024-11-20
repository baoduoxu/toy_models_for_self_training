import torch
import torch.nn as nn
import torch.optim as optim

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
d=3
mu = [2,-1,1]
dim = len(mu)
n_labelled = 2*d # O(d)
n_unlabelled = 2*d*10**4 # O(d*eps^{-2}) 0.01
delta = 0.9
eta = 0.01
T = 3000
B = 100
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
# 生成贝叶斯最优分类器附近的 test_data
test_data = dataset_gen.generate_test_data_near_bayes_optimal_classifier(n_test=20, eps=0.01)
# 按照 label 给 test_data 排序


# 使用无标签数据进行自训练
self_trainer = SelfTraining(eta=eta, sigma=sigma, B=B, T=T, test_data=test_data)
final_classifier, entropy_list = self_trainer.train(unlabelled_data, initial_classifiers[0])

self_trainer.plot_entropy(entropy_list, test_data)
# print(analyze_entropy_trend(entropy_list))
print(f'test data: {test_data}')

# 用 final_classifier 对 test_data 进行分类, 并与真实值对比
pred = torch.sign(torch.matmul(torch.Tensor(test_data[0]), final_classifier))
print(f'Predictions: {pred}')
print(f'True labels: {test_data[1]}')


print(final_classifier)
print(f'Bayes optimal classifier: {dataset_gen.mu / torch.norm(dataset_gen.mu)}, distance between the final classifier and the Bayes optimal classifier: {torch.norm(final_classifier - dataset_gen.mu/torch.norm(dataset_gen.mu))}')