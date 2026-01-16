"""
张量 -> 存储同一类型元素的容器

torch.tensor() 函数可以从多种数据结构创建张量， 只能是数值和布尔类型
torch.Tensor() 也是类似的功能, 但支持根据形状创建张量, 并且默认数据类型为 float32

torch.ones(), torch.ones_like() 创建全1张量
torch.zeros(), torch.zeros_like() 创建全0张量
torch.full(), torch.full_like() 创建指定值张量

创建线性和随机张量:
torch.arange() 创建等差数列张量  参数1: start, 参数2: end, 参数3: 步长
torch.linspace() 创建等间隔数列张量 参数1: start, 参数2: end, 参数3: 元素个数
torch.rand() 创建均匀分布随机数张量
torch.randn() 创建标准正态分布随机数张量
torch.randint() 创建指定范围内的随机整数张量
torch.random.initial_seed() 设置随机数种子, 保证实验可复现
torch.manual_seed() 设置随机数种子, 保证实验可复现


张量类型转换
张量的数据类型可以通过 .type(torch.int16) 方法进行转换

张量与 NumPy 数组互转
张量 -> NumPy 数组:
    使用 .numpy() 方法  共享内存
    使用 .numpy.copy() 方法  不共享内存
    使用 .ndarray() 方法  不共享内存

NumPy 数组 -> 张量:
    numpy nd数组
    使用 torch.tensor(nd数组) 函数  不共享内存
    使用 torch.from_numpy(nd数组) 函数  共享内存

标量张量 -> 数值:
    使用 .item() 方法

张量的点乘 和 矩阵乘法
点乘:
    要求：两个张量形状相同
    使用: torch.mul() 函数 或 * 运算符
矩阵乘法:
    要求: 第一个张量的最后一维大小等于第二个张量
    使用: torch.matmul() 函数 或 @ 运算符

张量的运算函数
    sum(), mean(), max(), min()   -> 都有 dim 参数, 指定维度进行运算, 0表示列, 1表示行, 默认对所有元素进行运算
    pow() / **, sqrt(), exp(), log(), log10(), log2()

张量的广播机制
当两个张量形状不同时, 可以通过广播机制进行运算
广播机制的规则:
1. 如果两个张量的维度数量不同, 形状较小的张量会在最左边补1, 直到两个张量的维度数量相同
2. 如果两个张量在某个维度上的大小不同, 则将该维度上较小的张量的大小扩展为较大的张量的大小, 扩展方式为复制元素
3. 如果在某个维度上, 两个张量的大小都不相同, 且其中一个张量的该维度大小不为1, 则会引发错误

张量的索引操作
分类:
1. 简单行列索引
2. 列表索引
3. 范围索引
4. 布尔索引
5. 多维索引

张量的形状操作
1. reshape() / view()  改变张量形状, 返回新张量, 原张量不变 ,
view无法使用的情况: 原张量不是连续存储的, 即经过transpose或permute操作后的张量,需要使用.contiguous() 方法先将其变为连续存储的张量
2. squeeze()  去除张量中所有为1的维度
3. unsqueeze()  在指定位置插入为1的维度
4. permute()  维度置换, 改变张量维度顺序
5. transpose()  交换指定的两个维度
6. flatten()  将多维张量展平为一维张量
7. contiguous()  将非连续存储的张量变为连续存储的张量 is_contiguous() 判断张量是否为连续存储

张量的拼接与拆分
拼接:
    torch.cat()  按指定维度拼接张量， 所有张量在非拼接维度形状必须相同，即除了拼接维度外其他维度形状相同
    torch.stack()  在指定维度堆叠张量, 新增一个维度，所有张量形状必须相同
拆分:
    torch.split()  按指定大小拆分张量
    torch.chunk()  按指定块数拆分张量

"""

import torch
import  numpy as np

def dm01():
    # 场景1: 标量 张量
    t1 = torch.tensor(5)
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 50)

    # 场景2: 二维列表 -> 矩阵 张量
    data = [[1, 2], [3, 4]]
    t2 = torch.tensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('-' * 50)

    # 场景3: 从 NumPy 数组 -> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.tensor(data)
    print(f't3: {t3}, type: {type(t3)}')
    print('-' * 50)

    # 场景4: 创建指定形状和数据类型的张量
    t4 = torch.Tensor(2, 3)  # 默认数据类型为 float
    print(f't4: {t4}, type: {type(t4)}')
    print('-' * 50)

    # torch.manual_seed()

def dm02():
    torch.manual_seed(42)
    t1 = torch.randint(1, 10, (5, 5))
    print(f't1:\n{t1}')
    # 场景1 : 简单行列索引
    print(t1[0, 0])  # 第一行第一列元素
    print(t1[:, 1])  # 第二列所有元素
    print('-' * 50)

    # 场景2: 列表索引
    rows = [0, 2, 4]
    cols = [1, 3, 4]
    print(t1[rows, cols])  # 取出 (0,1), (2,3), (4,4) 元素
    print('-' * 50)

    # 场景3: 范围索引
    # 左闭右开
    print(t1[1:4, 2:5])  # 取出第2到4行, 第3到5列的子矩阵
    print(t1[1::2, 0::2]) # 取出奇数行，偶数列, 步长为2
    print('-' * 50)

    # 场景4: 布尔索引
    mask = t1 > 5
    print(f'mask:\n{mask}')
    print(t1[mask])  # 取出所有大于5的元素
    print(t1[:, 2][t1[:, 2] > 5]) # 取出第3列大于5的元素
    print(t1[t1[:, 2] > 5]) # 取出第3列大于5的行数据
    print('-' * 50)

    # 场景5: 多维索引
    t2 = torch.randint(1, 10, (3, 4, 5)) # 3个4x5矩阵
    print(f't2:\n{t2}')
    print(t2[0, :, :])  # 取出第1个矩阵
    print(t2[:, 1, :])  # 取出所有矩阵的第2行
    print(t2[:, :, 2])  # 取出所有矩阵的第3列
    print(t2[1, 2, 3])  # 取出第2个矩阵的第3行第4列元素
    print('-' * 50)

if __name__ == "__main__":
    # dm01()
    dm02()