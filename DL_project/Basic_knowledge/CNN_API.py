import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def demo1():
    img = plt.imread("data/img.png")
    print(img.shape) # (640, 640, 4)
    # plt.imshow(img)
    # plt.show()

    # 卷积层 参数1：输入通道数 参数2：输出通道数，即kernel数量 参数3：卷积核大小 参数4：步长 参数5：填充
    conv = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=0)

    # tensor -> HWC -> CHW -> NCHW    N 是 batch size，即样本数量
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    print("input tensor shape:", img_tensor.shape)
    out = conv(img_tensor) # 计算公式: (W−K+2P)/S+1
    print("output tensor shape:", out.shape)

    # 查看特征图
    out_img = out.squeeze(0).permute(1, 2, 0).detach().numpy()
    print("output image shape:", out_img.shape) # (638, 638, 4)
    plt.imshow(out_img[:, :, 0]) # 只看第一个通道
    plt.show()

def demo2():
    inputs = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).float() # shape: (1, 3, 3) 多通道可改 (3, 3, 3)

    pool = nn.MaxPool2d(2, 1, 0) # kernel_size=2, stride=1, padding=0
    out = pool(inputs)
    print("最大池化:", out)

    pool = nn.AvgPool2d(2, 1, 0)
    out = pool(inputs)
    print("平均池化:", out)


if __name__ == '__main__':
    # demo1()
    demo2()