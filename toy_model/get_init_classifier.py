import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegressionSGD:
    def __init__(self, input_dim, delta, eta, T):
        self.delta = delta
        self.eta = eta
        self.T = T
        self.num_classifiers = 4 * int(torch.ceil(torch.log(torch.tensor(1 / delta))))
        self.classifiers = [torch.zeros(input_dim, requires_grad=True) for _ in range(self.num_classifiers)]
        
    def train(self, data, labels):
        for i in range(self.num_classifiers):
            for t in range(self.T):
                idx = torch.randint(0, data.size(0), (1,))
                x_t = data[idx]
                y_t = labels[idx]
                # print(f'self.classifiers[i]: {self.classifiers[i]}')
                pred = torch.dot(x_t.squeeze(), self.classifiers[i])
                loss = torch.log(1 + torch.exp(-y_t * pred))  # Logistic loss
                loss.backward()
                with torch.no_grad():
                    self.classifiers[i] -= self.eta * self.classifiers[i].grad
                    self.classifiers[i].grad.zero_()
        return self.classifiers
