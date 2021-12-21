# 【paper-note8】啥是Attention?

> 论文题目：《Attention Is All You Need》
>
> 论文作者：Ashish Vaswani Google Brain
>
> 收录：NIPs 2017



## 前言

还记得18年去南大参加MLA的时候，会上的大佬们都在说Attention mechanism，那么啥是Attention？简单点来说，Attention机制就是加权，目前其实现形式包括三种，我把它归纳成：

1. 基于CNN的Attention
2. 基于RNN的Attention
3. self-Attention，即Transformer结构

## Attention in CNN

其中基于CNN的Attention可以分为通道域和空间域的，分别可以去看SE-Block [1]和CBAM-Block [2]，其他的多数是这两个的变种。这里简单说一下，比如通道域，在某层的feature map有64个通道，则对每个通道赋一个权重，即$\hat{f_i} = a_i * f_i$，其中$a_i$表示每个通道的权重，$f_i$表示每个通道的原始特征，$\hat{f_i}$表示每个通道加权后的特征，而权重$a_i$是从原始所有特征中用小型神经网络算出来的，可以认为权重能够自动捕获通道间的依赖关系。

## Attention in RNN

理解了上面的CNN Attention，后面的都好办了，因为都是大同小异的，基于RNN的Attention也是如此，这里用文章 [3]的公式来解释一下，其使用了encoder-decoder结构，在decoder层加入attention结构：
$$
S_t = f(S_{t-1}, y_{t-1}, c_t)\\
c_t= \sum_{j=1}^{T_x} a_{tj} h_j
$$
可以看出是用$a_{tj}$ 对$h_j$ 进行加权，其中$a_{tj}$ 表示t时刻j个隐藏层的权重，公式如下：
$$
a_{tj} = \frac{exp(e_{tj})}{\sum_{k=1}^{T_x} exp(e_{tk})}
$$
熟悉的同学一眼就能看出这是个softmax，$e_{tj}$表示当前时刻decoder的输入$h_j$ 和t-1时刻的decoder的输出$S_{t-1}$的关联程度，关联程度越高，则该$h_j$ 的权重越大，公式如下：
$$
e_{tj} = g(S_{t-1}, h_j) = V\cdot tanh(W\cdot h_j + U\cdot S_{t-1} + b)
$$

## Self-Attention

上面两种情况稍微提一下，不做展开，有兴趣的同学可以去参考文献仔细看，本文着重要讲的是《Attention is all your need》的Transformer结构，也就是经常能听到的self-attention，该结构最初是用在机器翻译领域中，论文中说到，提出该方法的motivation是当使用RNN进行序列传导建模的时候，其本质是串联的，即 $h_t$ 的输出必须等待 $h_{t-1}$ 的输入，导致计算效率很低，不能进行并行计算。而Transformer直接把整个原始序列输入，不需要等待该，可以直接进行并行计算。

### Transformer框架

Transformer用了encoder-decoder结构，看下面的图就能了解大概框架，其中`encoder`结构由N层堆叠而成，每个层包含两个sub-layer，一个MHA（Multi-Head Attention）和一个全连接网络，每个sub-layer都用残差结构连接起来。输入是整个原始序列的嵌入，输出是$d_{model}$的向量。

`decoder`结构也由N层堆叠而成，每个层包含三个sub-layer，两个MHA和一个全连接层，基本和encoder类似，输入是原始输出向量的嵌入（因为不能让当前输出和后面的输出产生attention，论文中说是`prevent leftward information flow`，故要把当前和后面所有的输出都mask掉，在MHA的softmax中这些值设为$-\infty$）。随后，经过一个全连接层和softmax层，输出当前时刻预测的probabilities。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91c2VyLWdvbGQtY2RuLnhpdHUuaW8vMjAyMC8yLzI3LzE3MDg3MjM4NDBmZTgxYjI?x-oss-process=image/format,png)


### Multi-Head Attention

要讲清楚Multi-Head Attention就要从单个Attention讲起，论文中把单个Attention叫做`Scaled Dot-Product Attention`，结构如下图左边：


![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91c2VyLWdvbGQtY2RuLnhpdHUuaW8vMjAyMC8yLzI3LzE3MDg3MjNjOTJmMTQxNDU?x-oss-process=image/format,png)

首先定义queries $Q$，keys $K$，values $V$，则单个Attention的公式如下：
$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
由此可见，softmax算出来的是一个权值，以此对V进行加权。那么自相似性是怎么体现的呢？从上面的Transformer结构所知，Q，K，V三个向量是同一个input。。则算出来的权重就是与query最相关的key影响最大，即input序列中与当前元素最相关的元素影响最大。

Multi-Head Attention如上图右边所示，就是重复多次单个Attention再拼接输出向量，传给一个全连接层输出最终结果。公式如下：
$$
MultiHead(Q, K, V) = Concat(head_1, \dots , head_h)W^O\\
where\ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

至此，transformer的结构已经阐述完毕，我们发现这种结构的确能提高计算效率和捕获数据里的自相似性，而且能很好的处理长程依赖（因为输入是把所有元素一起输入，这里感叹一句谷歌爸爸真的有钱，没有足够的计算资源撑腰谁能想得出这种烧钱方法），里面的实现细节有很多有意思的地方，等我深挖一下，以后如果有机会在写篇博客说说。

## 参考文献

[1] Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141.

[2] Woo S, Park J, Lee J Y, et al. Cbam: Convolutional block attention module[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 3-19.

[3] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.