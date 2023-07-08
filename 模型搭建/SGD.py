import numpy as np

# 初始化模型参数
weights = np.zeros(num_features)
bias = 0

# 设置超参数
learning_rate = 0.01
num_epochs = 100
batch_size = 32

# 迭代更新模型参数
for epoch in range(num_epochs):
    # 打乱训练集顺序
    np.random.shuffle(train_data)
    # 分成迷你批次
    for batch in range(0, num_train_samples, batch_size):
        # 获取当前迷你批次
        X_batch, y_batch = train_data[batch:batch+batch_size, :-1], train_data[batch:batch+batch_size, -1]
        # 计算损失函数梯度
        grad_w = np.dot(X_batch.T, (y_batch - np.dot(X_batch, weights) - bias)) / batch_size
        grad_b = np.mean(y_batch - np.dot(X_batch, weights) - bias)
        # 更新模型参数
        weights += learning_rate * grad_w
        bias += learning_rate * grad_b
