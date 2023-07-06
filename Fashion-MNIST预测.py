# 对上面定义的Classifier类进行实例化
model = Classifier()

# 定义损失函数为交叉熵
criterion = nn.CrossEntropyLoss()

# 优化方法为SGD，学习率为0.003
optimizer = optim.SGD(model.parameters(), lr=0.003)

# 对训练集的全部数据学习15遍
epochs = 15

# 将每次训练的训练误差和测试误差存储在这两个列表里，供后面绘制误差变化折线图用
train_losses, test_losses = [], []

def train():
    print('开始训练：')
    for e in range(epochs):
        running_loss = 0

        # 对训练集中的所有图片都过一遍
        for images, labels in trainloader:
            # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
            optimizer.zero_grad()

            # 对64张图片进行推断，计算损失函数，反向传播优化权重，将损失求和
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 每次学完一遍数据集，都进行以下测试操作
        else:
            test_loss = 0
            accuracy = 0
            # 测试的时候不需要开自动求导和反向传播
            with torch.no_grad():
                # 将模型转换为评估模式，在该模式下不会影响到训练
                model.eval()

                # 对测试集中的所有图片都过一遍
                for images, labels in testloader:
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    # 等号右边为每一批64张测试图片中预测正确的占比
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    # 恢复Droput
            model.train()
            # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图
            train_losses.append(running_loss / len(trainloader))
            test_losses.append(test_loss / len(testloader))

            print("训练集学习次数：{}/{}..".format(e + 1, epochs),
                  "训练误差：{:.3f}..".format(running_loss / len(trainloader)),
                  "测试误差:{:.3f}..".format(test_loss / len(testloader)),
                  "模型分类准确性：{:.3f}".format(accuracy / len(testloader)))
