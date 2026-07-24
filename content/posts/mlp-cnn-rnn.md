---
title: 深度学习基础：MLP、CNN 与 RNN框架
date: 2026-07-24
slug: "mlp-cnn-rnn"
categories:
  - Deep Learning
tags:
  - MLP
  - CNN
  - RNN
  - Neural Network
---
引言：学习过MLP（多层感知机），CNN（卷积神经网络），RNN（循环神经网络）三个框架后，可以发现，三个模型本质上都是在做：输入数据X -> 特征提取 -> 分类器 ->  输出
不同的地方在于特征的提取的办法不同
接下来，我会用MPL模型做MINIST灰度手写图片分类，用CNN模型做CIFAR10图片分类，用RNN做情感分类为例，来说明三个模型的各自的用处和不同点

## MLP
### 架构：
Input X -> Linear1 ->Activation1 ->Linear2 ->Activation2 -> ...... ->Linear  -> Softmax ->Output $\hat y$

### 每一个epoch，MLP做：
1.把输入的数据展成一维向量，
2.全连接层做线性变换：$Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$ ，每次把上一层的输出作为下一层的输入，其中$A^{[0]}=X$ 
3.通过激活函数引入非线性 ：$A^{[l]}=g^{[l]}(Z^{[l]})$ ，因为线性变化的组合仍然是线性变换
4.全连接层线性变换和激活重复做，本质就是在做数学上的函数嵌套
5.完成输出层的线性计算，得到logits（原始打分）
6.通过softmax把logits转换成概率，选取概率最大的类别
7.计算损失，反向传播计算梯度
8.根据梯度更新参数


### MINIST灰度手写图片分类任务
MINIST灰度手写图片，单通道，大小为$28 \times28$ ,展开成列向量就是$784 \times 1$ ,参数量比较小，适合使用MLP模型来完成
![MINIST灰度手写图片](/deeplearning/mlpMINIST.png)
把上面每一个epoch做的任务详细写：
1.使用mini-bacth的方法，每次取64张MINIST图片作训练，每张图片展开成$784 \times 1$的列向量
2.通过全连接层做特征提取，其中由矩阵乘法可知，权重矩阵W 为$n^{[l]} \times n^{[l-1]}$，偏置矩阵b是$784\times 1$，其中$n^{[l]}$表示第l层的神经元的个数，我提供的代码，一共三个全连接层，第一层128个神经元，第二层64个神经元，第三层10个神经元
3.通过ReLU函数激活，$a=\max(0,z)$ ，当z>0,a=z;当z<=0,a=0
4.把上一层的输出作为下一层的输入，做该层的线性变换，然后激活，重复上述全连接和激活
5.进行输出层的线性变换，得到$64\times 10$ 的logits矩阵，10个数字就有10个原始打分
6.通过softmax回归分类，将logits转换成概率，公式是$a_i= \frac{e^{z_i}} {\sum_{j=1}^{n}e^{z_j}}$,  其中 $e^{z_i}$ 表示这个类别的得分，$0<a_i<1$ ,$\sum_i a_i=1$ ，选择10个类别中概率最大的类别
7.计算交叉熵损失$L(y,\hat y) = -y\log(\hat y) -(1-y)\log(1-\hat y)$ ,反向传播得到参数梯度，dW和db
根据梯度，更新参数

在更新参数上，采用Adam优化
随机梯度下降更新参数用的是 ：$W=W-\alpha dW$和$b=b-\alpha db$
但是如果参数中的大小相差较大时就容易在梯度下降的过程中发生震荡，而且学习率$\alpha$ 也不好统一调整，而Adam优化通过计算更新方向和更新大小的方式可以有效缓解这样的问题
这里用最简单的语言介绍这个优化算法
更新方向：
不再只关注当前的数据，而是结合前一段时间的历史趋势和当前趋势来更新方向
它给最近的数据更高权重，给过去的数据逐渐降低权重，从而使得更新的曲线更平滑
公式：$\boxed{ v_t=\beta v_{t-1}+(1-\beta)\theta_t }$
$\beta v_{t-1}$ 表示过去的平均  ，$(1-\beta)\theta_t$ 当前数据
假设：$v_0=0$
第一次：$v_1=0.1\theta_1$
第二次：$v_2 = 0.9v_1+0.1\theta_2$
代入：$=0.9(0.1\theta_1)+0.1\theta_2$
得到：$v_2= 0.09\theta_1+0.1\theta_2$
第三次：$v_3= 0.9v_2+0.1\theta_3$
展开：$= 0.081\theta_1 +0.09\theta_2 +0.1\theta_3$
权重：0.1  ,  0.09  ,  0.081  越来越小。
也就是过去数据影响指数下降 
指数加权平均数的权重衰减：第k天前$\boxed{ \beta^k(1-\beta) }$
定义有效记忆长度  $\frac{1}{e}$ ≈0.3679
$\beta$ ≈1时，k表示记忆长度≈$\frac1{1-\beta}$ 
因为$v_0=0$，初期训练数据时，得到的$v_i$ 和真实梯度相差较大，为了缓解这样的问题，还要引入偏差修正
公式：$\boxed{ v_t^{corrected} = \frac{v_t}{1-\beta^t} }$
随着t增大，$v_t$ 会快速趋近0，影响逐渐消失
高曲率方向：梯度方向反复变化 → 历史和当前抵消（梯度一正一负） → 减少震荡
低曲率方向：梯度方向长期一致 → 历史不断累积（梯度同正通负） → 加快移动

更新大小：
根据不同方向梯度的大小，自动调整学习率，梯度大的方向减小步长，梯度小的方向增大步长
$s_{dw}^{(t)} = \beta_2s_{dw}^{(t-1)} + (1-\beta_2)(dw^{(t)})^2$ ，用$dw$的平方消去正负影响和衡量梯度变化的剧烈程度
梯度$dw$除以$\sqrt{s_{dw}}$,,，梯度大的方向学习率调小，小的方向学习率调大，就能达到自动调整学习率的效果，加上很小的数$\epsilon$ ,是为了防止分母为零
更新：$\boxed{ W=W-\alpha \frac{dW}{\sqrt{s_{dw}}+\epsilon} }$

Adam优化得到新的跟新方向和更新大小，将两者结合起来
完整流程：
初始化$v_{dw}=0,\quad s_{dw}=0$
1.计算梯度 dW，db
2.梯度平均Momentum  $v_{dw} = \beta_1v_{dw} + (1-\beta_1)dW$
3.梯度平方平均RMSProp $s_{dw} = \beta_2s_{dw} + (1-\beta_2)(dW)^2$
4.修正偏差   $v_{dw}^{correct} = \frac{v_{dw}} {1-\beta_1^t}$和$s_{dw}^{correct} = \frac{s_{dw}} {1-\beta_2^t}$
5.最终更新参数 $W = W-\alpha \frac{v_{dw}^{correct}} {\sqrt{s_{dw}^{correct}}+\epsilon}$
其中超参数
学习率$\alpha=0.001$
Momentum参数$\beta_1=0.9$
RMSProp参数$\beta_2=0.999$
$\epsilon=10^{-8}$

8.根据得到的梯度更新参数
### MLP代码
下面给出ChatGPT生成的MLP代码
```python
import torch  
import torch.nn as nn  
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
  
  
# ==========================  
# 1. 加载 MNIST 数据  
# ==========================  
  
transform = transforms.ToTensor()  
#数据预处理  
#尺寸 28*28-->1*28*28
train_data = datasets.MNIST(  
    root="./data",  
    train=True,  
    download=True,  
    transform=transform  
)  
#得到60000个（1,28,28）图片和6000个label  
  
#测试集  
test_data = datasets.MNIST(  
    root="./data",  
    train=False,  
    download=True,  
    transform=transform  
)  
  
  
# batch  
train_loader = DataLoader(  
    train_data,  
    batch_size=64,#一次输入64*1*28*28  
    shuffle=True#打乱  
)  
  
  
test_loader = DataLoader(  
    test_data,  
    batch_size=64  
)  
  
  
  
# ==========================  
# 2. 定义 MLP# ==========================  
  
class MLP(nn.Module):#继承  
  
    def __init__(self):
        super().__init__()
  
        self.fc1 = nn.Linear(28*28,128)#MINIST图片展开成784为 输出128个神经元  
#self.fc1 = 一个全连接层  
#自动创建了一个128*784的权重矩阵W和128*1的偏置矩阵b  
#自动加入参数管理  
#forward计算z=WX+b ,自动支持梯度  
  
        self.fc2 = nn.Linear(128,64)  
  
        self.fc3 = nn.Linear(64,10)  
  
#定义前向传播  
    def forward(self,x):  
  
        # (batch,1,28,28)  
        # ->        # (batch,784)  
        x=x.view(x.size(0),-1)#x.size(0)取第0维，就是batch size，-1让pytorch自动计算  
#Tensor(batch数量, 通道数, 高度, 宽度)  
  
        x=torch.relu(self.fc1(x))#第一层 先\(Z^{[1]}=W^{[1]}X+b^{[1]}\) 然后ReLU  
  
        x=torch.relu(self.fc2(x))  
  
        x=self.fc3(x)#输出层 得到64*10的logits  
        #不要加softmax，因为CrossEntropyLoss里面已经包含  
  
  
        return x  
  
  
  
model=MLP()  
  
  
  
# ==========================  
# 3. loss 和 optimizer# ==========================  
  
loss_fn=nn.CrossEntropyLoss()#计算误差  
#使用softmax把logits转成概率  
#计算负对数似然  
#等价  
#logits = model(x)  
#prob = softmax(logits)  
#loss = CrossEntropyLoss(prob,y)  
  
#优化器  Adam优化  
optimizer=torch.optim.Adam(  
    model.parameters(),#拿到所有参数  
    #创建Adam管理器，保留每个参数的历史梯度和更新状态  
    # 向后传播，计算梯度  
    #跟新参数  
    lr=0.001  
)  
  
  
  
# ==========================  
# 4. 训练  
# ==========================  
  
epochs=5  
  
  
for epoch in range(epochs):  
  
    total_loss=0  
  
  
    for images,labels in train_loader:  
  
  
        # 前向传播  
  
        pred=model(images)#得到64*10  
  
  
        # loss  
        loss=loss_fn(pred,labels) 
  
  
  
        # 反向传播  
  
        optimizer.zero_grad()#pytorch默认梯度累积，so 每次更新前清零  
  
        loss.backward()  
  
        optimizer.step()#更新参数  
  
  
  
        total_loss+=loss.item()#累加损失  
  
  
  
    print(  
        f"Epoch:{epoch+1}, loss:{total_loss:.4f}"  
    )  
  
  
  
# ==========================  
# 5. 测试  
# ==========================  
  
correct=0  
total=0  
  
  
with torch.no_grad():#测试关闭梯度  
  
    for images,labels in test_loader:  
  
  
        output=model(images)#预测  
  
  
        prediction=output.argmax(dim=1)#最大类别  
  
  
        correct += (prediction==labels).sum().item()#预测正确数量  
  
        total += labels.size(0)  
  
  
  
print(  
    f"Accuracy:{correct/total*100:.2f}%"#计算准确率  
)  
  
  
  
# ==========================  
# 6. 看几个预测  
# ==========================  
  
images,labels=next(iter(test_loader))  
  
  
output=model(images)  
  
prediction=output.argmax(dim=1)  
  
  
print("预测:")  
print(prediction[:10])  
  
  
print("真实:")  
print(labels[:10])
```


## CNN

### 架构：
输入X -> 特征处理器 -> 分类器 ->输出
其中特征处理器中，重复经过：卷积层 -> 激活层 -> 池化层
分类器：展平 -> 全连接 -> 输出层

### 与MLP比较，CNN更适合处理图像
刚刚提到，MLP是一开始就将数据展成一维向量，对于$28\times 28$ 这样的MINIST灰度手写图片或许还好，但是对于大尺寸的RGB图片，展开成一维向量之后参数量太大了 
而且图像原本是具有局部相关性和空间结构的，如果用MLP直接把图像展平的话，也会破快这种局部相关性和空间结构
而CNN不一样，CNN看图像：卷积核提取局部特征，特征图保留空间响应，池化扩大有效视野，多层堆叠完成抽象组合，最终由分类器输出图像类别

### CIFAR10图片分类任务

依旧采用mini-batch，每次处理64张图片，每张RGB图片三个通道，大小$32\times32$
每张图片就是$3\times 32\times32$的
#### 特征处理器：
##### 卷积层：把局部模式变成特征图
不展平图像、不做全连接的情况下，从图像中提取有意义的局部特征
三个东西：卷积核、stride移动步幅，padding
卷积核：一个滑动窗口，用于提取局部特征，每次将窗口范围的图像数据与对应位置的卷积核参数相乘再求和，得到一个响应值
（图片来源于B站up主谦行Aling）
![CNN卷积核结构](/deeplearning/CNNjj.png)
卷积核画完整张图像后，把响应值按空间位置关系排列，就得到这个卷积核对应的特征图
卷积核尺寸：每次看多大的局部区域,常见${3}\times 3$和$5\times 5$，我的例子中用的是${3}\times 3$
对于多通道的图像
![CNN卷积核结构](/deeplearning/CNNdtd.png)
卷积核数=特征图数  
一个卷积核在每个通道上分别做乘加，再把结构加在一起，得到一个响应值
输出通道数=卷积核数
每个卷积核独立扫描输入，各自输出一张特征图，n张特征图叠在一起，输出$n\times H \times W$
stride移动步幅：卷积核每次移动几个像素
一般卷积层为1，池化层或者下采样为2
padding：给图像边缘补0
${3}\times 3$ 卷积核在图像内部完整滑动，输出尺寸会比输入小
图像边缘补0后，提取局部特征的同时保持空间尺寸不变反复Conv -> ReLu -> Pool

##### 激活层ReLU：让特征组合具备非线性

##### 池化层：降低分辨率，扩大有效视野，网络逐渐有能力看向整体
使用一个小窗口在特征图上滑动，对空间维度做下采样，常见最大池化
我的例子中取$2\times 2$池化窗口，就是每次窗口中的4个值取最大值，可以理解为用四个值中的最大值作为这个窗口的代表，从而压缩图片
这样一次池化后，图片被压缩为原来的二分之一，又是用${3}\times 3$的卷积核来在压缩后的图片中提取特征，就能达到扩大感受野的效果


### 分类器：展平->全连接->输出层
和上面的MLP差不多
Flatten：把特征图拉成一维向量
Linear：映射到类别数维度
Softmax：转成概率

### 代码
依旧是ChatGPT生成的代码
~~~python
import torch  
import torch.nn as nn  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader  
  
  
# ==========================  
# 数据  
# ==========================  
  
transform = transforms.ToTensor()#数据处理，把图片变成3*32*32  
  
#加载训练集  50000张图片和10个类别  
train_data = datasets.CIFAR10(  
    root="./data",  
    train=True,  
    download=True,  
    transform=transform  
)  
  
  
test_data = datasets.CIFAR10(  
    root="./data",  
    train=False,  
    download=True,  
    transform=transform  
)  
  
#一次去64张图片  
train_loader = DataLoader(  
    train_data,  
    batch_size=64,  
    shuffle=True  
)  
#images.shape  64*3*32*32  
  
test_loader = DataLoader(  
    test_data,  
    batch_size=64  
)  
  
  
  
# ==========================  
# CNN  
# ==========================  
  
class CNN(nn.Module):#定义神经网络，继承pytorch网络基类  
  
    def __init__(self):  
  
        super().__init__()  
  
  
        # 输入:  
        # (batch,3,32,32)  
        self.conv1 = nn.Conv2d(  
            in_channels=3,#输入通道3  
            out_channels=16,#输出通道16  16个特征图  
            kernel_size=3,#卷积核大小为3  
            padding=1#边缘补一圈  
        )  
#输出  (64,16,32,32)  
        self.pool = nn.MaxPool2d(2)#最大池化  
        #池化敞口大小为2*2  (64,16,16,16)  
  
  
        self.conv2 = nn.Conv2d(  
            in_channels=16,#这里的输入通道数为上一层的输出通道数  
            out_channels=32,  
            kernel_size=3,  
            padding=1  
        )  
        #(64,16,16,16)->(64,32,16,16)  
  
  
# 两次池化:  
#  
# 32×32  
# ↓ pool  
# 16×16  
# ↓ pool  
# 8×8  
#  
# 通道32  
#  
# 展开:  
# 32*8*8
  
        self.fc = nn.Linear(#全连接层  
            32*8*8,#输入特征数  
            10#输出10个类别  
        )  
  
  
    def forward(self,x):  
  
  
        # 第一层卷积  
  
        x = torch.relu(  
            self.conv1(x)  
        )  
        #卷积得到（64,16,32,32） 然后ReLU，负数变为0  
  
  
        x = self.pool(x)#（64,16,16,16）  
  
  
  
        # 第二层卷积  
  
        x = torch.relu(  
            self.conv2(x)  
        )  
        #（64,32,16,16）  
  
  
        x = self.pool(x)#（64,32,8,8）  
  
  
  
        # 展平（64,2048）  
  
        x=x.view(  
            x.size(0),  
            -1  
        )  
  
  
        x=self.fc(x)#全连接 计算WX^T+b,得到（64,10）  
  
  
        return x  
  
  
  
  
model=CNN()  
  
  
  
# ==========================  
# loss + optimizer  
# ==========================  
  
loss_fn=nn.CrossEntropyLoss()  
  
  
optimizer=torch.optim.Adam(  
    model.parameters(),  
    lr=0.001  
)  
  
  
  
# ==========================  
# 训练  
# ==========================  
  
epochs=5  
  
  
for epoch in range(epochs):  
  
    for images,labels in train_loader:  
  
  
        output=model(images)  
  
  
        loss=loss_fn(  
            output,  
            labels  
        )  
  
  
        optimizer.zero_grad()  
  
  
        loss.backward()  
  
  
        optimizer.step()  
  
  
  
    print(  
        "Epoch:",  
        epoch+1,  
        "loss:",  
        loss.item()  
    )  
  
  
  
# ==========================  
# 测试  
# ==========================  
  
correct=0  
total=0  
  
  
with torch.no_grad():  
  
    for images,labels in test_loader:  
  
  
        output=model(images)  
  
  
        prediction=output.argmax(dim=1)  
  
  
        correct += (  
            prediction==labels  
        ).sum().item()  
  
  
        total += labels.size(0)  
  
  
  
print(  
    "Accuracy:",  
    correct/total  
)
~~~
演示效果：
![[Pasted image 20260724133724.png|526]]

## RNN循环神经网络
这里用many-to-one的情感分类任务做例子
### 架构：
输入X -> 特征提取 -> 分类器
特征提取：文本预处理 -> 特征提取 -> 编码 ->文本表示
分类器：分类器： 全连接 -> 输出层

### 情感分类
#### 特征提取：
经历：文本 -> 分词 -> 编号 -> 词嵌入 -> RNN
以6句话为例：
"i love this movie",  
"this movie is great",  
"i like this film",  
  
"i hate this movie",  
"this movie is terrible",  
"i dislike this film"

去重之后，一共有10个单词，所以我们的词典库一会总共有10个词
为每个词创建一个独立的编号，我们用one-hot向量来表示
第i个词向量中第i个位置元素值为1，其余元素值为0
例如："i"的编号为0，$o_i=\begin{bmatrix} 1\\ 0\\0\\...\\0\end{bmatrix}$
"love"的编号为1，$o_{love}=\begin{bmatrix} 0\\1 \\0\\...\\0 \end{bmatrix}$
然后引入词嵌入，当前我的例子中，词典库只有10个词，所以只需要10个$10\times1$的one-hot向量就能表示所有词了，但是要是词典库有10 000或者100 000个词呢，
如果好用这样的方法表示，参数量就会很大，而且不能表示出词与词之间语义联系
但是我们知道，一个词通常是可以根据一些属性描述出来的
比方说 "apple"，是水果，不是人类，有甜度，是多汁的......
“Queen”，是人类，不是水果，是女性......
对一个10 000个词的词典，用300个词大概就可以描述出词典库里的所有词了
引入一个嵌入矩阵E，大小$10000 \times 300$ ,每行代表一个词向量，有300列，是不同的属性
每个词进行$o_i=E^To_i$的计算，因为$o_i$ 只有相应i位置的元素值为1，其余位置元素值都为0，所以，相当于每次取$E^T$的第i列作为词向量，$E^t$ 的列向量就是E的行向量
所以，词嵌入的时候要把词编号转换成词向量，只要每次取的词嵌入矩阵E的对应编号的第i行就可以了

然后到RNN
RNN将状态在自身网络循环传播，可以用于处理语音，文字，股票价格这类序列模型
模型引入一个隐藏状态$a^{<t>}$，用于把前面的信息按照序列保存下来，其中,t表示第t个时间步
![RNN前向传播](/deeplearning/rnn.png)

$a^{\langle t\rangle} = g(W_{aa}a^{\langle t-1\rangle} + W_{ax}x^{\langle t\rangle} + b_a)$
$\hat y^{\langle t\rangle} = g(W_{ya}a^{\langle t\rangle}+b_y)$
$a^{\langle t\rangle} = g( \text{当前输入贡献} + \text{历史记忆贡献} )$
$W_{aa}$ ：a->a表示隐藏状态到隐藏状态的权重
$W_{ax}$ ：x->a输入对隐藏状态的影响
$W_{ya}$ ：a->y隐藏状态产生输出
RNN 在不同时间步处理不同的输入 $x^{\langle t\rangle}$时，使用的是同一套参数 $W_{ax}, W_{aa}, W_{ya}$

就这样一直使用将前面的信息压缩保存到$a^{<t>}$，到以后就能能到存有完整一句话信息的$a^{<T_y>}$

#### 分类器：全连接->输出层
Linear：模型根据一句话给出预测分数
sigmoid：logit转成概率

### 代码：
依旧chatgpt生成的
~~~python
import torch  
import torch.nn as nn  
import torch.optim as optim #优化器  
  
  
# =========================  
# 1. 构造简单数据集  
# =========================  
  
sentences = [  
    "i love this movie",  
    "this movie is great",  
    "i like this film",  
  
    "i hate this movie",  
    "this movie is terrible",  
    "i dislike this film"  
]  
  
  
# 0:负面  1:正面  
labels = [  
    1,  
    1,  
    1,  
  
    0,  
    0,  
    0  
]  
  
  
# =========================  
# 2. 建立词典  
# =========================  
  
word_to_idx = {}#创建一个空字典  存：单词 -> 编号  
  
for sentence in sentences:  
    for word in sentence.split():  
#split 把字符串拆成单词  
        if word not in word_to_idx:  
            word_to_idx[word] = len(word_to_idx)  
  
  
print(word_to_idx)  
  
  
vocab_size = len(word_to_idx)  
  
  
# =========================  
# 3. 文本转换成数字  
# =========================  
  
def encode(sentence):#定义一个函数，一句话->数字序列  
  
    return [  
        word_to_idx[word]  
        for word in sentence.split()  
    ]  
  
  
X = [#对所有句子转换  
    encode(sentence)  
    for sentence in sentences  
]  
# X = [  
#     [0,1,2,3],  
#     [2,3,4,5],  
#     [0,6,2,7],  
#     [0,4,2,3],  
#     ...  
# ]  
  
  
print(X)  
  
  
# =========================  
# 4. 定义RNN模型  
# =========================  
  
class RNNClassifier(nn.Module):  
  
    def __init__(self):#初始化  
  
        super().__init__()  
  
  
        # embedding:  
        # 词编号 -> 词向量  
        self.embedding = nn.Embedding(  
            vocab_size,#10个词  
            8#每个词用8个数字表示  
            #pytorch内部创建了一个嵌入矩阵 10*8            #10行，每一行对应一个词  
            #8列，表示每个词的embedding维度  
  
            #查表：输入编号i，直接取E的第i行，得到8*1的列向量  
  
        )  
  
  
        # 创建一个RNN层  
        #1.创建了输入权重w_{ax}  16*8  
        #2.创建隐藏状态权重w_{ax}  16*16  
        #3.创建偏置b  16*1  
        #4.设置输入格式  
        self.rnn = nn.RNN(  
            input_size=8,#每个词输入8位的embedding  
            hidden_size=16,#隐藏状态大小  
            batch_first=True#输入格式  (batch,sequence,length)            #一次处理多少个样本b，每句话多少个词，每个词多少维  
        )  
  
  
        # 分类层  
        #一个全连接\(z=Wa+b\)  
        #得到logit  
        self.fc = nn.Linear(  
            16,#隐藏状态16维  
            1#二分类只输出一个数字  
        )  
  
  
    def forward(self, x):  
  
        # x:  
        # (batch, seq_len)  
  
        # 词嵌入  
        x = self.embedding(x)  
  
  
        #  
        # (batch, seq_len, 8)        #  
  
        output, hidden = self.rnn(x)#\(a^t=f(Wa^{t-1}+Wx^t+b)\)  
        #由当前输入和上一个隐藏状态，output保存每一个时间步的隐藏状态  
        #hidden只保存最后一个状态的隐藏状态  
  
        # hidden:  
        # 最后一个时间步的隐藏状态  
        #  
        # shape:        # (1,batch,16)  
  
        hidden = hidden.squeeze(0)#去掉第一维  
        #pytorch规定的hidden形状：(层数, 句子数量, 隐藏维度)  
        #，squeeze（0）：如果第0维大小是1，就删除这一维  
        # 这个代码只有一层rnn  
        #（1，batch，16）->（batch，16）  
  
  
        # (batch,16)  
  
  
        result = self.fc(hidden)#分类 \(z=Wa+b\)  读出一个分数  
  
  
        return result  
  
  
  
model = RNNClassifier()#创建对象  
  
  
# =========================  
# 5. 损失和优化器  
# =========================  
  
  
loss_fn = nn.BCEWithLogitsLoss()#计算二分类任务的损失  
#包括sigmoid计算  
  
  
optimizer = optim.Adam(  
    model.parameters(),  
    lr=0.01  
)  
  
  
  
# =========================  
# 6. 训练  
# =========================  
  
  
for epoch in range(100):  
  
  
    total_loss = 0  
  
  
    for sentence, label in zip(X, labels):  
#C++里的pair  
  
        # 转tensor  
        x = torch.tensor(  
            sentence  
        ).unsqueeze(0)  
#拿到的sentence = [0, 1, 2, 3]是list  
#转换成pytorch能处理的tensor([0,1,2,3])  
#.unsqueeze(0) 在指定位置第0维增加一个维度  
  
        # (1,seq_len)  
  
#把标签转换成pytorch的浮点tensor，变成也就是一个长度为1的向量  
        y = torch.tensor(  
            [label],  
            dtype=torch.float  
        )  
  
  
        # 前向传播  
  
        output = model(x)  
  
  
        output = output.squeeze(1)#删除大小为1的维度  
  
  
#\(L=-y\log(\hat y)-(1-y)\log(1-\hat y)\)  
        loss = loss_fn(  
            output,  
            y  
        )  
#模型根据一句话给出预测分数  
#把多余的维度去掉，让形状和标签一致  
#比较预测和真实答案，得到误差  
  
        # 清梯度  
  
        optimizer.zero_grad()  
  
  
        # 反向传播  
  
        loss.backward()  
  
  
        # 更新参数  
  
        optimizer.step()  
  
  
        total_loss += loss.item()  
  
  
  
    if epoch % 10 == 0:  
  
        print(  
            "epoch:",  
            epoch,  
            "loss:",  
            total_loss  
        )  
  
  
  
# =========================  
# 7. 测试  
# =========================  
  
  
def predict(sentence):  
  
  
    model.eval()#切换到评估模式  不要再计算当前batch统计量，用训练好的统计值。  
  
  
    x = torch.tensor(  
        encode(sentence)  
    ).unsqueeze(0)  
  
  
    with torch.no_grad():  
  
        output = model(x)  
  
  
        prob = torch.sigmoid(output)  
  
  
    if prob.item()>0.5:  
  
        print(  
            sentence,  
            "=> positive",  
            prob.item()  
        )  
  
    else:  
  
        print(  
            sentence,  
            "=> negative",  
            prob.item()  
        )  
  
  
  
predict(  
    "i love this film"  
)  
  
  
predict(  
    "i hate this movie"  
)
~~~


以上就是我对MLP，CNN，RNN三种模型架构的理解，如有错误，欢迎指出！