# 【paper-note7】Several Papers About Video Classification

## Abstract

最近看了点视频分析的论文，归纳总结一下，分别是如下4篇。

1. 《Large-scale video classification with convolutional neural networks》Andrej Karpathy `CVPR2014`
2. 《Two-stream convolutional networks for action recognition in videos》Karen Simonyan `NIPS2014`
3. 《Learning spatiotemporal features with 3D convolutional networks》Du Tran `ICCV2015`
4. 《Real-world Anomaly Detection in Surveillance Videos Waqas》Waqas Sultani `CVPR2018`

前三篇是视频分类领域，最后一篇属于监控视频异常检测领域。顺带一提，以前的论文笔记长篇大论，比较耗时耗精力，以后要转变一下风格，写简明论文笔记，把核心思想记录下来就可以了。

## 0. video classification

视频分析和图像最大的不同在于视频多了一个时间维度，如何利用时间维度提高性能降低复杂度是视频分析方法研究中关注比较多的点。很多相似任务，比如分类，检测，分割等，直接用图像的方法逐帧检测当然可行，比如SSD直接在视频上逐帧检测还是可以接受的，但是没有把时间信息利用起来。同时也会引入图像中本来没有的任务，比如temporal action detection要界定时间框，比如目标跟踪，除了检测之外，还要逐帧识别出检测的是同一个东西。

今天要讲的视频分类就是视频分析中最基础的任务，比较general的来看，现在的视频分类方法先从一个视频中采样出一堆clips，比如从一个5分钟的视频抽出不重叠的100个clips，每个clips是10帧，然后把每个clip当作输入，输出就是视频的类别标签。由此看出，这个任务并不需要处理长程依赖问题，但是除了clip-level的feature，有些研究者意识到video-level的feature也很重要，这就需要用RNN把local的clip-level represent 和global的video-level represent结合起来，这是后话了。

## 1. Slow-Fusion CNN

斯坦福李飞飞组的工作，可以把它当作入门video的第一篇paper，探索性很强的一篇paper，为以后的工作提供了很多参考。

### Contribution

* 首先是提供了一个数据集：Sports-1M，在youtube上面采集的一百万+视频，共有487个类别，在这个数据集之前，做视频分类研究用的数据集典型代表有UCF-101和HMDB-51，虽然UCF-101挺大了，有13k个视频，但是对于深度学习来说，样本还是太少，Sports-1M的出现缓解了这个问题。

* 其次，探索了不同分辨率组成的CNN和不同的时间信息（motion信息）连接形式，横向比较网络的性能，找到最好的架构。最后发现，高低分辨率双流网络和slow-fusion的方式带来的效果是最好的。

  网络结构也比较简单，主要借鉴了AlexNet，如下：

  ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200102192723.png)

  ​	Figure 1 Slow Fusion Network

  * fovea stream是原视频的center crop，context stream是原视频分辨率缩小两倍之后的低分辨率视频。这么做的原因是减小输入尺寸，加快训练速度。用fovea的另一个原因是直觉上，拍视频的时候会把拍摄物体放在画面中央，对画面做中心裁切能减少边角的干扰。这个操作有点像是给中心加权。

  ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200102195842.png)

  ​	Figure 2 approaches for fusing information over temporal dimension

  * 此外，在获取时间维度信息上面，采用了slow fusion的操作，就是把10 frames的clips通过三层网络逐渐融合到一起，以此学习到运动信息。

### Result

原文做了很多实验，这里讲一个，用Feature Histograms + Neural Net当作baseline，探索要不要学习时间信息和要如何融合时间信息这两个问题。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200102200608.png)

1. 要不要时间信息，这个比较好理解，要时间信息就是用fusion的方法，不要时间信息就是用单帧作为输入。Figure 2 最左边就是单帧作为输入，直接丢弃时间维度信息。针对单帧得到context stream和fovea stream两个网络。
2. 如何融合时间信息，可以看Figure 2 剩下的图，针对三种融合方式和平均三种融合方式分别做了实验。

从结果来看，单帧图片用上双流的方法（这里的双流指的是context stream和fovea stream）得到的效果已经足够好了，和slow fusion的效果只差了一点点，所以作者就说，视频分类中时间维度的fusion好像没啥用。



## 2. Two Stream

 牛津大学在NIPs2014年的工作，双流法因为效果出众也成为现在视频分析领域的主流方法之一，缺点是要用brox的方法计算光流，即使用上opencv GPU版本，计算一个clips的光流仍然需要0.6s，效率不高。双流法包含空间流和时间流两个网络，下面做简单阐述。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104154202.png)

### Spatial Stream

空间流用了从视频中随机采样单一帧作为输入得到softmax分类，卷积网络用的也类似于AlexNet，由于UCF-101和HMDB-51的数据集太少，对于训练图像分类器不够，因此作者采用了ImageNet ILSVRC-2012的预训练网络，然后再fine-tune，单一图片分类器就能再UCF-101上得到**73.0%**的效果。

### Temporal Stream

时间流把224\*224\*2L作为输入，L就是clips大小，2L是光流的x和y轴方向上的位移量，经过实验最终L=10。由于brox方法很慢，所以论文作者先用opencv把光流跑出来保存下来，用int8类型保存能减少存储量，再把跑好的光流作为输入传给卷积网络，最后得到softmax值，两个softmax值用SVM方法融合起来，得到最后的输出。

另外，作者还探索了保存同样像素点的光流和保存轨迹光流的区别，最后发现保存轨迹没什么效果提升，具体可以去看论文。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104155535.png)

### Results

效果还是很出色的，已经超过IDT了，视频分析中用上光流的效果都挺好的，因为这样能够很好handle住运动信息，最终报告平均accu是88%，具体指标如下图。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104155804.png)



## 3. C3D

Facebook在CVPR2015的工作，也算是视频分析3D卷积网络的开端，想法非常简单，用3D卷积代替2D卷积，网络架构还是和AlexNet类似，这里感叹一句，几年前的神经网络都是AlexNet风格啊。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104160107.png)

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104165158.png)

### contributions

C3D做了一些超参数的探索，最终确定用3x3x3的卷积核，stride是1x1x1，max pooling用2x2x2。

输入是先把视频缩放到UCF-101一半的大小：128x171，然后再抽出不重叠的16-帧clips，得到3x16x128x171，最后进行random crop得到3x16x112x112的输入。

在Sports-1M上训练，因为Sports-1M的视频很长，作者对每个视频随机抽取5个2秒的clips，然后对每个clip用上面的方法得到3x16x112x112作为输入。同时也用了I380K的预训练模型在Sports-1M做fine-tune。在Sports-1M的Clip hit@1是46.1%，Video hit@1是61.1%。

最后3个C3D网络测试UCF-101上的准确度，分别是在I380K上训练的网络，用Sports-1M上训练的网络，I380K训练然后用Sports-1Mfine-tune的网络，需要注意的是，没有在UCF-101上做fine-tune，直接用fc6的特征加上一个简单的线性SVM，结果依然很好。

### Results

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104164609.png)

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104165119.png)

可以看出C3D相对于two-stream的优势在于效率高，而且结果也不差。

另外，作者还做了紧凑性研究，用PCA对fc6的特征进行将为，并配合一个简单的线性SVM，效果比IDT好了一大截。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104165418.png)

最后，用t-SNE算法对特征进行降维可视化，很酷炫。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104165550.png)

## 4. Real-word Anomaly Detection in Surveillance Videos

这篇论文和前三篇没什么关系，CVPR2018上面的工作，做的是监控视频异常检测领域，用了弱监督学习的思想，简单的说，就是只有视频级别的二分类标签（异常OR正常），然后能够学习到异常发生在哪个segment。

### contributions

首先，论文提出了一个异常监控视频的数据集，从youtube上面采集的，共13种真实异常事件，并给了异常事件时间上的标注。共1900个视频，平均每个视频有7247帧，大概5分钟，共128小时。

**核心思想**也比较简单，把视频切分成32个segments放入bag中，定义bag中没有一个是异常segments视频就算是正常，bag中有一个是异常segment视频就算异常。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104172302.png)

网络的输入如下：先把一个视频resize到240x320，固定帧率为30fps，切分出不重复的32个segments，然后对于单个segment的每16-frames clip用C3D提取特征，平均所有的clips的features得到单个segment的feature，最后用1个神经元做2分类。object function是SVM的改进版，适应弱监督学习，叫做MIL Ranking Loss with sparsity and smoothness constraints。

### MIL Ranking Loss with sparsity and smoothness constraints

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104172952.png)

SVM是关键，可以从上式看出用了一个max自动选择出影响最大的segments。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104173118.png)

总目标函数如上，其中f就上面的线性函数，正则化项1代表时间平滑度，代表相邻的segments应该有相近的score，正则化项2代表稀疏性，真实世界中的异常事件应该很少且很短，每个视频只有1个或少数的异常。

### result

正确率达到了75.41%，效果还挺好的，在弱监督下面已经很不错了。误判率也只有1.9%

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104173644.png)

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104173759.png)

从跟groud-truth的对比也可以看出，随着迭代次数增多，网络确实学习到了异常事件发生的时间。其实这个操作在泛化到图像领域能够得到一些启发。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20200104174509.png)

最后，由于数据集包含13个异常动作的标注，也可以拿来尝试动作分类任务，作者用C3D和TCNN在整个视频上针对每个16-frames跑，提出的特征用近邻分类器分类，效果非常差，C3D只有23%，TCNN只有28.4%。原因是这个数据集是未裁剪的原始数据集，并且类内差异非常大。









