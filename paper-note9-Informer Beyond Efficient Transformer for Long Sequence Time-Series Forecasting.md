# 【paper-note7】Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

> 论文作者：Haoyi Zhou Beihang University
>
> 收录：AAAI2021 best paper

## 前言

Transformer模型在NLP领域提出之后，风靡众多领域，有大一统的模式。这篇论文就是对 transformer 进行改进，提出 Informer 模型，应用在长时序预测领域。该模型设计合理精巧，利用长尾分布截取头部的attention点积对，提高运行效率、降低运行成本；同时模型结合长时序预测的特点，在输出阶段采用生成式的方法，一次性生成待预测的序列，避免了传统序列预测方法中串行生成的缺点。





















看这篇paper主要是在查阅关于ISP AI化[2]的时候有人提到了谷歌的这篇多帧降噪论文，其指出直接用神经网络代替传统ISP的pipline不现实，因此曲线救国，把isp中的一部分任务交给神经网络来处理，诸如降噪，提升动态范围，以及细节恢复等问题有可能被神经网络代替。

> 二、是否使用一个神经网络替代ISP
>
> 这个答案其实是不确认的，但是所有人都认为应该至少部分难以一般算法很难得到更好的结果和调试结果部分如降噪，提升动态范围，以及细节的恢复，或者是图像融合的部分是最有可能被神经网络的方法代替。这方面建议大家可以看下google的一片论文。其中将Burst降噪多帧合成处理过程使用神经网络实现了。

## 核心思想

本文针对连续图像序列降噪做出改进，运用了KPN进行降噪核预测，对图像序列中的每一张图片都预测一个滤波器，将滤波器作用于图像序列的每张图上，最后对整个图像序列取平均得到输出$\hat{Y}$ ，以此达到对齐和降噪的目的。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190721170228.png)

## 主要贡献

1. 将后处理图像（就是普通sRGB图像）数据转化为带有raw特性的数据，其实就是估计噪声水平，这样，每张图片都带有噪声水平信息了。
2. 网络能够预测独立的核，与源图像做内积之后能直接输出最终图像，即**直接合成像素**。并且能够查看图像序列中的哪张图像是怎么用的。
3. 在微小的不对齐下，网络可以预测核函数。
4. 证明了有噪声水平信息的输入到网络中比没有噪声信息的训练效果要好。

## 具体内容

### 问题描述

目标：用一个手持相机获取包含N张图片的图片序列，{$X_1, X_2, ..., X_N$} , 从中选择一张图片作为参考图片（不要求是第一张），利用剩余的图片对参考图片进行降噪。

### 噪声水平估计[1]

- shot noise：Poisson分布，$\mu$和$\sigma^2$等于原始信号值$y_p$
- read noise：Gaussian分布

观察值$x_p$：![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190722125311.png)

$\sigma_r$和$\sigma_s$对于每张图片都是固定的，可以通过相机的传感器增益来改变（ISO）。

### 合成数据集

用了Open Image dataset[3]，对单张图片上面加了局部偏移和全局偏移，并从已有训练数据的噪声参数中均匀采样参数，按上式把噪声加到数据集上。

在这之前，还做了两步，一是逆转gamma校正，产生近似线性颜色空间的patches。二是在[0.1, 1]之间随机采样，线性放缩数据，压缩直方图强度，让他更接近真实数据。这点在HDR+ [4] 的isp pipline 中是通过欠曝来避免高光损失的。

### 模型

- 输出图片：线性平均

  ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190722131013.png)

- 噪声标准差作为网络输入：

  ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190722131112.png)

- 损失函数

  - 基础损失函数：在像素强度上用L2损失，像素梯度上用L1损失。

    - ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190722131314.png)

    - 这里不用直接的gamma校正函数$X^\gamma$是因为，当$X$接近于0时，梯度会变的无穷大。

  - 退火损失函数：发现只有参考图片的滤波是非零的，网络进入了局部极小值。

    - ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190722131705.png)

    - 这里$\beta$和$0 < \alpha < 1$是超参数。t是迭代次数，$\alpha = 0.9998, \beta = 100$，当$\beta \alpha^t \gg 1$时，后一项鼓励每一个滤波器独立地先对自己对应的图像进行对齐和降噪，当t趋于无穷时，后一项就消失了。

至此，本文主要思想以阐述完毕，感兴趣的同学可以下载论文详细查阅。

## 参考文献

[1] G. Healey and R. Kondepudy. Radiometric CCD camera calibration and noise estimation. TPAMI, 1994.

[2] [不止是去噪-从去噪看AI ISP的趋势](https://cloud.tencent.com/developer/news/278554)

[3] I. Krasin, T. Duerig, N. Alldrin, V. Ferrari, S. Abu-El-Haija, A. Kuznetsova, H. Rom, J. Uijlings, S. Popov, A. Veit, S. Belongie, V. Gomes, A. Gupta, C. Sun, G. Chechik, D. Cai, Z. Feng, D. Narayanan, and K. Murphy. Openimages: A public dataset for large-scale multi-label and multi-class image classification. Dataset available from https://github.com/openimages, 2017.

[4] S. W. Hasinoff, D. Sharlet, R. Geiss, A. Adams, J. T. Barron, F. Kainz, J. Chen, and M. Levoy. Burst photography for high dynamic range and low-light imaging on mobile cameras. SIGGRAPH Asia, 2016.

## 补充材料

噪声模型的部分推导过程，详细的可见文献[1]。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190721164328.png)