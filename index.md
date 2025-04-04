---
title: Home
layout: home
layout: test1
layout: test2
---

This is a *bare-minimum* template to create a Jekyll site that uses the [Just the Docs] theme. You can easily set the created site to be published on [GitHub Pages] – the [README] file explains how to do that, along with other details.

If [Jekyll] is installed on your computer, you can also build and preview the created site *locally*. This lets you test changes before committing them, and avoids waiting for GitHub Pages.[^1] And you will be able to deploy your local build to a different platform than GitHub Pages.

More specifically, the created site:

- uses a gem-based approach, i.e. uses a `Gemfile` and loads the `just-the-docs` gem
- uses the [GitHub Pages / Actions workflow] to build and publish the site on GitHub Pages

Other than that, you're free to customize sites that you create with this template, however you like. You can easily change the versions of `just-the-docs` and Jekyll it uses, as well as adding further plugins.

[Browse our documentation][Just the Docs] to learn more about how to use this theme.

To get started with creating a site, simply:

1. click "[use this template]" to create a GitHub repository
2. go to Settings > Pages > Build and deployment > Source, and select GitHub Actions
# 深度学习

## 第1章-第3章  基础的内容

1. 手动梯度归零

- 批量计算梯度的时候，需要在for循环中手动的将梯度归零。
- ![image-20250120161339452](images/image-20250120161339452.png)

2. 一个简单的手动实现的线性回归

   ```python
   lr = 0.03
   num_epochs = 3
   net = linreg
   loss = squared_loss
   
   for epoch in range(num_epochs):
       for X, y in data_iter(batch_size, features, labels):
           l = loss(net(X, w, b), y)  # X和y的小批量损失
           # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
           # 并以此计算关于[w,b]的梯度
           l.sum().backward()
           sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
       with torch.no_grad():
           train_l = loss(net(features, w, b), labels)
           print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
   ```

   这里需要注意，linreg、squared_loss、sgd这里是自己手动编写的，当然实际现在是不需要这么写的。

   - 解释为何 ==l.sum().backward()==：

     - 在 PyTorch 中，`backward()` 函数用于计算损失相对于模型参数的梯度。然而，`backward()` 需要一个标量（即单个数值）作为输入来开始反向传播过程。这是因为梯度的概念是针对单个输出值相对于各个输入值的变化率而言的。

       当你有多个损失值时（例如，在一个小批量中有多个样本，每个样本都有自己的损失），你实际上有一个形状为 `(batch_size, ...)` 的张量，而不是一个单独的标量值。为了从这个张量中获得一个单一的标量值，你可以对所有元素求和或者取平均值。这也就是为什么你会看到 `.sum()` 或者 `.mean()` 被调用的原因。

       具体来说：

       - **`.sum()`**：它将整个小批量的损失加起来得到一个总的损失值。这是你代码片段中的做法。
       - **`.mean()`**：它会对小批量中的损失取平均值，这也是一个常见的做法，特别是在计算交叉熵损失等场景下。

       选择使用 `.sum()` 或者 `.mean()` 取决于你的需要以及你如何调整学习率。如果你使用了 `.mean()`，那么梯度的大小不会依赖于小批量的大小，这有时会让训练过程更加稳定。但是，使用 `.sum()` 也是完全可以接受的，只要你确保学习率设置得当，可以适应不同大小的小批量。

       因此，`l.sum().backward()` 这一行代码的作用就是将小批量中的所有样本的损失加总成一个标量，然后基于这个总损失进行反向传播，以计算出关于模型参数的梯度。这样做的结果是，你得到了一个能够代表整个小批量的梯度信息，进而可以用来更新模型参数。

     - 但是==注意==，这是手动的，如果是使用from torch import nn，那么是自动帮你求好了sum（），你只需要直接backward就可以。

       - ```python
         num_epochs = 3
         for epoch in range(num_epochs):
             for X, y in data_iter:
                 l = loss(net(X) ,y)
                 trainer.zero_grad()
                 l.backward()
                 trainer.step()  # 这个是模型的更新，更新参数w 和 b
             l = loss(net(features), labels)
             print(f'epoch {epoch + 1}, loss {l:f}')
         ```

   3. 手动编写的思想：

      - 其实就是先手动定义了一个函数f(x)  = wx + b,这样是知道w与b的。会模拟出很多的值
      - 接下来才是神经网络的工作，采用一定的策略（比如从正态分布中随机取w与b作为初始的参数），然后经过训练，反向传播，更新参数，最后是可以得到最终的模拟的w与b的最终值。

   4. ==读取小批量数据==

      - shuffle=True 表示打乱顺序取，但是还是会取完所有的内容。同时注意，一般来说，**训练数据**需要打乱顺序，而测试不需要。

      - ```python
        batch_size = 256
        
        def get_dataloader_workers():  #@save
            """使用4个进程来读取数据"""
            return 4
        
        train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                     num_workers=get_dataloader_workers())
        # 我们看一下读取训练数据所需的时间。
        timer = d2l.Timer()
        for X, y in train_iter:
            continue
        f'{timer.stop():.2f} sec'
        ```

      - 这里看读取的时间。需要注意，你读取的速度，肯定要大于训练的速度吧。


## 4 多层感知机

### 4.1 感知机的基础知识

1. 普通的感知机就是二分类问题

   1. 有d个输入，输出是单一的输出

   1. ![image-20250122223201746](images/image-20250122223201746.png)

   1. 如果激活函数是线性的，可以看到 hence之后的内容，还是一个线性的。所有你加入隐藏层，其实还是效果还是一个单层的感知机。

2. 激活函数为何不用之前那个简单的函数？

   1. ==激活函数的作用==是用来避免层数的塌陷，所有最后一层是不加激活函数的！
   2. ![image-20250122223519276](images/image-20250122223519276.png)
   3. 因为你看，如果是之前那个1  0  在x=0的时候，是不好求导的，所有梯度下降这一步是不好做的。同理，tanh也是一个软的  ，就是为了求导的方便。
   4. Relu = max(x,0)，就是因为不是能线性的，所有将＜0的地方换掉了，这就不是线性了。其实就是老东西新整，重新起了个名字而已。

3. 多类分类：softmax其实就是将数据全部投影到0-1之间，然后满足归一化的特性，当然投影的数据解释为==概率==，其实就是在softmax中加入了一层隐藏层

   1. 经验：

      - ![image-20250122224921328](images/image-20250122224921328.png)

      - 如果是多隐藏层，m1可以128，m2＜m1，比如取个64，m3<m2,比如取个32。就是最好是慢慢的压缩数据，不断地慢慢提炼信息。所以在感知机中，可以先放大，在缩小。当然cnn是先压缩，在放大，防止过拟合。

4. 总结：

   1. ![image-20250122225246608](images/image-20250122225246608.png)

### 4.2模型选择与过拟合

![image-20250207193818049](images/image-20250207193818049.png)

- ==注意，测试数据集相当于“高考",只能用一次，不可以用来调整超参数！！！==
- <u>**验证数据集**</u>是用来调整参数的，相当于月考呀。一般有时候代码偷懒，就写成了test_data
  - 所有理论上来说，验证数据集的精度不是真正代表你在新数据上的泛化能力
- **<u>训练数据集</u>**相当于课后作业，有答案。用来训练模型参数的
- 训练误差是training dataset，而泛化误差是testing dataset .当然，testing dataset 一般指的是 vaidation dataset。用泛化误差来进行参数的调整.
- 假设我有100w个数据，那么我用80w做训练+验证，用20w做测试。其中80w的训练+验证选取**5-则交叉验证* **，也就是每次训练用80/5=16w的数据做验证，剩下64w的数据做训练。

###### K-则交叉验证

- 当数据不太多的时候。<font color='yellow'>一般深度学习如果数据集太多，这个方法就没必要使用了。</font>

- 在训练过程中的。

- ![](images/image-20250207194554394.png)

- 这里蓝色的演示的是三折验证的情况。一般取5，是指**算法会迭代运行5次**，每次选择其中的**一个子集作为验证集**，而**剩余的4个子集合并起来作为训练集**。这意味着在整个**交叉验证过程中的每一次迭代，都会使用到全部的数据**，但每次使用的训练集和验证集是不同的。

  - 如何与epoch结合呢？：

    - 在机器学习中，**epoch**和**交叉验证（如5折交叉验证）**是两个独立但可以结合使用以优化模型训练过程的概念。

      - **Epoch**：指的是整个训练集被算法完整地遍历一次的过程。在一个epoch中，模型通过反向传播等方法调整其权重以减少损失函数。通常，为了获得较好的性能，模型需要进行多个epoch的训练。

      - **5折交叉验证**：用于评估模型性能的一种技术，它将数据集分成5个子集或“折”。然后重复进行5次训练和验证过程，每次使用不同的一个子集作为验证集，其余4个子集作为训练集。这样可以更全面地了解模型在不同数据划分情况下的表现，并帮助防止过拟合。

      ### 如何搭配使用

      <font color = 'yellow'>当你使用5折交叉验证时，对于每一个“折”（即每一次迭代），你都会执行一个完整的训练过程，这个过程中你可以选择运行多个epoch</font>。具体来说：

      1. **数据分割**：首先，将你的数据集分为5个折。
      2. **循环开始**：针对每个折：
         - 将其中的一个折作为验证集，其余四个折合并为训练集。
         - 使用这四个折组成的训练集对模型进行训练，这里你可以设定训练多少个epoch。
         - 在选定的epoch数后，使用该轮未参与训练的验证集来评估模型的性能。
      3. **结果汇总**：完成所有折的训练和验证后，平均各个折上的模型性能指标，得到模型的整体估计性能。

      这种组合方式可以帮助你不仅更好地评估模型的泛化能力，而且可以通过调整epoch的数量找到最佳的训练参数，避免过度拟合或欠拟合。需要注意的是，在实际操作中，有时也会先确定合适的epoch数目（例如通过初步实验或使用验证集），然后再应用交叉验证评估模型的整体性能。

2. 过拟合与欠拟合
   - ![](images/image-20250207201716307.png)

   - ![image-20250208170549200](images/image-20250208170549200.png)

- 核心任务：
  - 1)蓝色的点往下走  2)红色的范围变小 
- 所以有时候为了让泛化误差往下降，不得不接受一定程度的过拟合
- vc了解一下就行
- ![image-20250207202548877](images/image-20250207202548877.png)
- ![image-20250207202603700](images/image-20250207202603700.png)
- 数据复杂度：多个重要因素：（样本个数、每个样本的原色个数、时间空间结构、多样性）

###### QA

- 问题8：验证数据集和训练数据集的数据清洗（如异常值处理）和特征构建（如标准化）是否需要放在一起处理？
  - 两种方法。一种是一起处理，一种是只处理训练的，将训练的均值与方差用在验证中的。
  - 一般来说，用第一种更好一些，鲁棒性更强。
- 问题17:超参数怎么设置？如何调整？
  - 专家设置。老中医
  - 随机选取一个组合，随机选100个。推荐用这种。或者自己手动调整。
- 问题18：假设我做一个二分类问题，实际情况是1/9的比例，我的训练集一两种类型的比例应该是1/1还是1/9？
  - 沐神建议：如果数据集不够好，验证数据集要分布好  55 开 ；要不然可能出现，全部判多的。精度达到90%。但是不符合要求。可以加权矩阵来避免这个事情
  - 弹幕建议：可以用F1-score来测评；要看AUC值了；
- 问题19：k折交叉验证的目的是确定超参数吗？然后还要用这个超参数再训练一遍全数数据吗？
  - 是的。第一个是k折确定好，然后重新训练好。第二个是直接找选定超参数后K折里精度最好（或随便）的一折，选择该模型，不再重新训练；第三种，k个模型都拿下来，真的做预测的时候，将测试数据集，让k个都预测一遍，然后取均值。（代价是k倍）

### 4.3 正则化

- 视频：https://www.bilibili.com/video/BV1Z44y147xA/?spm_id_from=333.788.top_right_bar_window_history.content.click&vd_source=cb88355d0fa4b837438e420a28b4f3b2

- 定义从拉格朗日方向理解：凡是可以减少过拟合的方法，都叫做正则化。
- w叫做权重，b叫做偏移。  
- L1和L2正则化是针对w的。
  - 正则化后的最值点与本来的最值点是有差距的，但是差距不是很大
  - ![image-20250209114822757](images/image-20250209114822757.png)
  - L2范数：
    - ![image-20250209113321032](images/image-20250209113321032.png)
  - L1范数：
    - ![image-20250209113338288](images/image-20250209113338288.png)

- 根据正则化，w和b不同时，使的损失函数最小时，w和b的绝对值不同。  因为你训练的时候使用的不是全集。

- 这是在训练时一般，但是在真正测试的时候，大参数会使得结果出现很大的问题。

- 想法：<font color = 'yellow'>限制w的可行域范围。</font>

- 从权重衰减角度：

  - ![image-20250209115533434](images/image-20250209115533434.png)

  - 可以看出来，W前面会更新
  - ![image-20250209115612229](images/image-20250209115612229.png)

  - 我们希望，可以减少这种弯弯绕绕，得到类似于蓝色曲线这种，泛化性更好，这类似于人为的规定限制。（控制f泰勒展开的高阶项）

### 4.6 Dropout

- 丢弃法是在层与层之间的 ，对每个元素进行扰动
  - ![image-20250209144249021](images/image-20250209144249021.png)
- 使用方法
  - ![image-20250209144639763](images/image-20250209144639763.png)
- 正则项：旨在训练中使用，在推理过程中，输出的是本身 h = dropout(h  )
- 总结
  - 丢弃法将一些输出项随机置0来控制模型复杂度
    常作用在多层感知机的隐藏层输出上
    丢弃概率是控制模型复杂度的超参数
- 一般来说，Nan是梯度的问题。

### 4.7 模型初始化和激活函数

- 1）我们为了防止出现梯度爆炸或者梯度消失，需要尽量保持每层的输出和梯度在一定的期望和方差值，而保持这样的期望和方差的一个方法就是对权重进行合理的初始化， 通过推导我们了解到我们需要保持一个很难的条件，而这个条件的近似是使用Xavier进行权重初始化，从而使得每层的输出和梯度尽量保持在一定的期望和方差，进而预防梯度爆炸。 

- 2）而对于激活函数，通过推导我们知道激活函数在原点附近需要保持类似σ=x，而对于tanh和relu这个是满足的，但是对于sigmoid则需要一定的处理，也就是4xsigmoid（x）-2。 也就是通过这两种方法可以大致解决梯度爆炸/消失等数组稳定性问题。

## 5 神经网络基础

### 5.1层与快

- ```python
  class MySequential(nn. Module): 
  def __init_(self, * args): 
  	super ()._init_()
  # *args：接受不定数量的参数（参数就是层或者块），* arg是python中的迭代器，说明这里面有多个参数依次被调用
  ```

- 可以自定义 _init__ 和forward()，反向求导是对forward自动进行求导

  - ```python
    class FixedHiddenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            # 不计算梯度的随机权重参数。因此其在训练期间保持不变
    		xxxxxx		
    
        def forward(self, X):
            X = self.linear(X)
            # 使用创建的常量参数以及relu和mm函数
            X = F.relu(torch.mm(X, self.rand_weight) + 1)
            # 复用全连接层。这相当于两个全连接层共享参数
            X = self.linear(X)
            # 控制流
            while X.abs().sum() > 1:
                X /= 2
            return X.sum()
    ```

### 5.2参数管理

已经定义了层，学会如何访问层中的**参数**

- 加下划线为原地操作，不加为复制后修改。例如，

  - ```python
    nn.init.normal_(m. weight, mean=0, std=0.01)#（替换函数）
    nn.init.zeros_(m.bias)
    ```

- net.apply(xxx)  表示将net中的所有block都传入xxx函数遍历一遍，具体干啥取决于xxx函数的功能。类似于for loop的功能



   # 实验步骤

### 实验2.1 VLAN配置步骤及具体命令  

**实验环境**：华为三层交换机（以用户提供的端口信息为例）  

---

#### **1. 创建VLAN并命名**  

- **创建VLAN 2和VLAN 3**：  

  ```  
  <Switch> system-view  
  [Switch] vlan batch 2 3  
  ```

---

#### **2. 将端口分配到VLAN**  

- **PC1连接端口`Ethernet 0/0/7`分配至VLAN2**：  

  ```  
  [Switch] interface Ethernet 0/0/7  
  [Switch-Ethernet0/0/7] port link-type access  
  [Switch-Ethernet0/0/7] port default vlan 2  
  [Switch-Ethernet0/0/7] quit  
  ```

- **PC2连接端口`Ethernet 0/0/10`分配至VLAN3**：  

  ```  
  [Switch] interface Ethernet 0/0/10  
  [Switch-Ethernet0/0/10] port link-type access  
  [Switch-Ethernet0/0/10] port default vlan 3  
  [Switch-Ethernet0/0/10] quit  
  ```

---

#### **3. 配置VLAN接口的IPv6地址（三层功能）**  

- **为VLAN2和VLAN3配置IPv6地址**：  

  ```  
  [Switch] interface Vlanif 2  
  [Switch-Vlanif2] ipv6 enable  
  [Switch-Vlanif2] ipv6 address 2001:db8:3::1/64  
  [Switch-Vlanif2] quit  
  
  [Switch] interface Vlanif 3  
  [Switch-Vlanif3] ipv6 enable  
  [Switch-Vlanif3] ipv6 address 2001:db8:1::2/64  
  [Switch-Vlanif3] quit  
  ```

---

#### **4. 验证配置**  

- **查看VLAN信息**：  

  ```  
  [Switch] display vlan  
  ```

  输出应显示：  

  - VLAN2包含端口`Ethernet0/0/7`  
  - VLAN3包含端口`Ethernet0/0/10`  

- **查看接口IPv6配置**：  

  ```  
  [Switch] display ipv6 interface Vlanif 2  
  [Switch] display ipv6 interface Vlanif 3  
  ```

---

#### **5. PC配置**  

- **PC1（VLAN2）**：  
  - IPv6地址：`2001:db8:1::1/64`  
  - 网关：`2001:db8:3::1`  
- **PC2（VLAN3）**：  
  - IPv6地址：`2001:db8:1::2/64`  
  - 网关：`2001:db8:2::1`  

---

#### **6. 连通性测试**  

- **同一VLAN内（PC1 ping网关）**：  

  ```  
  C:\> ping 2001:db8:1::1  
  ```

  **预期结果**：成功（Reply from 2001:db8:1::1）。  

- **不同VLAN间（PC1 ping PC2）**：  

  ```  
  C:\> ping 2001:db8:2::2  
  ```

  **预期结果**：成功（需三层交换机启用IPv6路由）。  

---

#### **7. 关键问题解答**  

- **默认VLAN数量**：  
  通过`display vlan`查看，默认VLAN为VLAN1，包含所有未分配的端口（如未使用的端口）。  

- **验证VLAN端口分配**：  
  使用`display vlan`确认`Ethernet0/0/7`属于VLAN2，`Ethernet0/0/10`属于VLAN3。  

- **MAC地址表查看**：  

  ```  
  [Switch] display mac-address  
  ```

  输出字段包括：`MAC地址、VLAN、端口、类型（动态/静态）`。  

---

#### **命令总结表**  

| 步骤             | 命令示例                                                     |
| ---------------- | ------------------------------------------------------------ |
| 创建VLAN         | `vlan batch 2 3`                                             |
| 命名VLAN         | `vlan 2` → `name VLAN2`                                      |
| 分配端口至VLAN2  | `interface Ethernet0/0/7` → `port link-type access` → `port default vlan 2` |
| 分配端口至VLAN3  | `interface Ethernet0/0/10` → `port link-type access` → `port default vlan 3` |
| 配置VLAN接口IPv6 | `interface Vlanif 2` → `ipv6 enable` → `ipv6 address 2001:db8:1::1/64` |
| 验证配置         | `display vlan`、`display ipv6 interface brief`               |

---

#### **注意事项**  

1. 确保交换机已启用IPv6路由功能（默认可能未启用）：  

   ```  
   [Switch] ipv6  
   ```

2. 若无法跨VLAN通信，检查：  

   - VLAN接口IPv6地址是否正确  
   - PC网关是否配置正确  
   - 防火墙或ACL是否拦截流量  

3. 保存配置防止丢失：  

   ```  
   [Switch] save  
   ```

If you want to maintain your docs in the `docs` directory of an existing project repo, see [Hosting your docs from an existing project repo](https://github.com/just-the-docs/just-the-docs-template/blob/main/README.md#hosting-your-docs-from-an-existing-project-repo) in the template README.

----

[^1]: [It can take up to 10 minutes for changes to your site to publish after you push the changes to GitHub](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll#creating-your-site).

[Just the Docs]: https://just-the-docs.github.io/just-the-docs/
[GitHub Pages]: https://docs.github.com/en/pages
[README]: https://github.com/just-the-docs/just-the-docs-template/blob/main/README.md
[Jekyll]: https://jekyllrb.com
[GitHub Pages / Actions workflow]: https://github.blog/changelog/2022-07-27-github-pages-custom-github-actions-workflows-beta/
[use this template]: https://github.com/just-the-docs/just-the-docs-template/generate

[^2]: [It can take up to 10 minutes for changes to your site to publish after you push the changes to GitHub](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll#creating-your-site).

[Jekyll]: https://jekyllrb.com
[Markdown]: https://daringfireball.net/projects/markdown/
[Liquid]: https://github.com/Shopify/liquid/wiki
[Front matter]: https://jekyllrb.com/docs/front-matter/
[Jekyll configuration]: https://jekyllrb.com/docs/configuration/
[source file for this page]: https://github.com/just-the-docs/just-the-docs/blob/main/index.md
[Just the Docs Template]: https://just-the-docs.github.io/just-the-docs-template/
[Just the Docs]: https://just-the-docs.com
[Just the Docs repo]: https://github.com/just-the-docs/just-the-docs
[Just the Docs README]: https://github.com/just-the-docs/just-the-docs/blob/main/README.md
[GitHub Pages]: https://pages.github.com/
[Template README]: https://github.com/just-the-docs/just-the-docs-template/blob/main/README.md
[GitHub Pages / Actions workflow]: https://github.blog/changelog/2022-07-27-github-pages-custom-github-actions-workflows-beta/
[customize]: {% link docs/customization.md %}
[use the template]: https://github.com/just-the-docs/just-the-docs-template/generate
