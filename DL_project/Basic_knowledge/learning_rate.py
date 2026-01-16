import torch
from torch import optim
import matplotlib.pyplot as plt

def func(x_t):
    return torch.pow(2 * x_t, 2)  # y = 4 * x^2

def plot_for_lr(lr, max_iteration=10):
    x = torch.tensor([2.0], requires_grad=True)
    iter_rec, loss_rec, x_rec = [], [], []

    for i in range(max_iteration):
        y = func(x)
        y.backward()
        iter_rec.append(i)
        loss_rec.append(y.item())
        x_rec.append(x.item())
        x.data.sub_(lr * x.grad)
        x.grad.zero_()

    return iter_rec, loss_rec, x_rec

def plot_results(lrs, max_iteration=10):
    rows = (len(lrs) + 1) // 2  # 计算行数
    fig, axs = plt.subplots(rows, 4, figsize=(16, 4 * rows))

    for idx, lr in enumerate(lrs):
        iter_rec, loss_rec, x_rec = plot_for_lr(lr, max_iteration)
        row, col = divmod(idx, 2)  # 计算当前学习率的行列索引

        # 左子图：迭代次数 vs 损失
        axs[row, col * 2].plot(iter_rec, loss_rec, '-ro')
        axs[row, col * 2].grid()
        axs[row, col * 2].set_xlabel("Iteration")
        axs[row, col * 2].set_ylabel("Loss")
        axs[row, col * 2].set_title(f"Learning Rate: {lr} (Iteration vs Loss)")

        # 右子图：函数曲线 + 下降轨迹
        x_t = torch.linspace(-3, 3, 100)
        y_t = func(x_t)
        axs[row, col * 2 + 1].plot(x_t.detach().numpy(), y_t.detach().numpy(), label="y = 4*x^2")
        y_rec = [func(torch.tensor(i)).item() for i in x_rec]
        axs[row, col * 2 + 1].plot(x_rec, y_rec, '-ro', label="Trajectory")
        axs[row, col * 2 + 1].grid()
        axs[row, col * 2 + 1].legend()
        axs[row, col * 2 + 1].set_xlabel("x")
        axs[row, col * 2 + 1].set_ylabel("y")
        axs[row, col * 2 + 1].set_title(f"Learning Rate: {lr} (Function and Trajectory)")

    plt.tight_layout()
    plt.show()


def demo():
    # 0.参数初始化
    LR = 0.1  # 设置学习率初始化值为0.1
    iteration = 10
    max_epoch = 200
    # 1 初始化参数
    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)
    # 2.优化器
    optimizer = optim.SGD([w], lr=LR, momentum=0.9)
    # 3.设置学习率下降策略
    gamma = 0.95
    # 等间隔生成学习率衰减
    # 调整方式：lr = lr * gamma
    # scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=gamma)

    # 指定间隔学习率衰减
    # 调整方式：lr = lr * gamma
    # scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 125, 160], gamma=gamma)

    # 指数学习率衰减
    # 调整方式：lr= lr∗ gamma ^ epoch
    # scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # 余弦学习率衰减
    # 调整方式：lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * T_cur / T_max)) / 2
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)

    # 4.获取学习率的值和当前的epoch
    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):
        lr_list.append(scheduler_lr.get_last_lr())
        epoch_list.append(epoch)
        for i in range(iteration):  # 遍历每一个batch数据
            loss = ((w*x-y_true)**2)/2.0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
        # 更新下一个epoch的学习率
        scheduler_lr.step()
    # 5.绘制学习率变化的曲线
    plt.plot(epoch_list, lr_list, label="Multi Step LR Scheduler",color="darkblue")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # lrs = [0.01, 0.1, 0.2, 0.3]
    # plot_results(lrs, max_iteration=10)
    demo()
