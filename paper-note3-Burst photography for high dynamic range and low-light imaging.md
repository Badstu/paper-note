# 【paper-note3】Burst photography for high dynamic range and low-light imaging on mobile cameras
> 论文题目：《Burst photography for high dynamic range and low-light imaging
> on mobile cameras》
>
> 论文作者：Samuel W. Hasinoff Google Research
>
> 收录：SIGGRAGH2016

# 前言

早在三四年前，小米手机五搭载了索尼IMX298摄像头，各种参数拿出来在当年可谓是一流，但是小米手机自带相机软件拍照效果不尽人意，于是国内数码玩家想尽各种办法，翻墙下载谷歌套件，移植谷歌相机app，然后再用谷歌相机拍的照和小米相机拍的照一对比，真的是天壤之别。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190804204548.png)

同一颗摄像头，只是换了个app就能得到完全不一样的效果，不得不感叹谷歌爸爸的实力，而谷歌相机搭载的牛逼成像算法，也正是这篇文章要讲到的谷歌`HDR+`。

# 核心思想

本文针对手机拍照提出了一个全新的成像算法，能够完全代替传统ISP pipline，并且在暗光环境和高动态范围场景中表现出色，

我们把HDR+系统概括为四点主要特点及贡献：

1. 首先，和以往采用包围曝光的方法实现HDR有着本质不同，HDR+采用**固定曝光**获取一系列图像序列，所谓包围曝光就是获取一堆不同曝光的图像，有些欠曝有些过曝，有些正常曝光，然后合成出高动态范围的图像；而HDR+首先聚焦在降噪上，因此采用固定曝光，即每张图片的曝光量都是相同的，以此来保证对齐具有更多鲁棒性。另外固定曝光是欠曝的，这点很好的避免高光溢出，增加动态范围。
2. 算法的输入是Bayer raw格式文件，不是常规意义上去马赛克后的RGB(YUV)文件，这点的好处是RGB图像由硬件ISP直接输出，并且有点像黑盒，会带来不必要的post-processing，直接处理raw格式能够存储更多的位深度和绕过不必要的色调映射和空间降噪。
3. 提出了基于快速傅里叶变换和2D/3D维纳滤波的降噪方法，能够加速降噪过程。
4. 最后，算法由Halide语言实现，可以运行在大多数消费级手机上，并且用户只需要按下快门就可以，整个过程不需要用户介入。对于一张12Mpix的图像，大概需要4秒时间成像。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190804211001.png)

上图是整个成像流程的示意图，整个流程分为两个部分：

上半部分是取景器的处理流程，对于取景器来说需要低像素的实时图像，这点传统ISP做的很好，虽然成像质量一般，但是实时性可以保障，这部分另外一个作用是决定曝光量，这里有一个自动曝光算法；

下半部分是多帧降噪的HDR+算法本体，大概流程就是首先由快门和传感器获取一系列拥有固定曝光的图像序列，需要注意的是，这些图像是欠曝的，为啥是欠曝，结合平常的摄影经验来说，数码相机宁欠勿曝，因为欠曝能后期拉回来，过曝就GG。其次对完整分辨率图像做对齐和合并操作，接下来就是调整白平衡，去马赛克，局部色调映射，全局色调映射，去雾，锐化，色调饱和度调整等常规操作，得到最终图像。

# 自动曝光算法

处理HDR场景包含三个步骤：

1. 故意的欠曝
2. 获取多帧图像序列
3. 压缩动态范围

欠曝已经成为处理HDR图像的常规操作了，可以看看文献[1]，其中文献[2, 3]把HDR任务当成降噪任务，也给本文提供了思路。

自动曝光算法在paper里面写的不是很清楚，这里给出简单的理解，可能有出入，欢迎指正，作者先花了好久用传统的包围曝光方法对5000个正常人生活中常见的场景拍照，得到HDR数据集，每个张图片都有短曝光和长曝光两个版本，当用户拍一张新的照片前，就在数据集里面找到相似的数据集，然后确定曝光时长。大概就是这个意思，作者把它叫做`Auto-exposure by example`。

# 对齐图像

## 参考图像选择

对齐图像就是把备选图像对齐到参考图像上，关于参考图像的选择，参考文献[4]，采用了一种基于raw输入的绿色通道上的梯度度量的方法，被称为lucky imaging。

## 处理raw image

虽然输入的是raw格式，但是在检测位移时，raw格式带有马赛克，不好做，然而去马赛克的复杂度又太大，所以干脆把raw格式的2X2 block直接变成灰度图，由此，12Mpix的raw图像就变成了3MPix的灰度图，在灰度图上面做对齐比在raw格式上面做对齐容易多了

## 分层对齐

分层对齐就是由粗粒度到细粒度的位移检测，然后应用对齐，具体操作就是最小化下式：

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190804215740.png)

$u$和$v$是x和y轴的偏移量，$u_0$和$v_0$是从上个粒度继承下来的偏移量。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190804215807.png)

其中最细的粒度是32*32pixel，其中最大的偏移量是64pixel。

需要注意的是，在粗粒度对齐检测时，用L2距离，因为这样能提升准确率，而在细粒度对齐检测时，用L1距离，因为在细粒度上，只存在像素级的偏移。

# 合并图像

本文设计了基于一对频域和时域上的滤波器的方法，并且作用在参考图像和备选图像的分块中，一般来说，分块大小是16 X 16，对于暗光场景，分块大小是32 X 32。

在设计滤波操作之前，本文又一次提到了噪声水平估计方法[5]，当得到噪声方差之后，就可以很轻易的分辨出`misalignment`和`noise`了，对于信号x，噪声方差$\sigma^2$可以被表示为Ax+B，并服从于泊松分布（均值是x）。A和B分别是模拟增益和数字增益。

最后用频域合并的方法合并图像：

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190804221630.png)

其中T(w)是二维傅里叶变换的结果，w=(w_x, w_y)代表空间频率，z是图像索引。

而A(w)是传统维纳滤波的一种变体：

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190804221737.png)![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190804221758.png)

这个式子这么解释，上式可以简化成$(1 - A_z) *T_z + A_z * T_0$，当备选图像是因为噪声产生的误差时，A_z会变小，T_z作用变大，T_0左右变大；当备选图像时因为位移产生误差，A_z会变大，T_z作用变小，T_0作用变大。这个式子主要解决鬼影问题：下图(b)人脑袋位置出现了白色横条，就是因为错把位移当成噪声造成的。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190804222439.png)

本篇论文笔记就到此，感兴趣的同学可以下载论文阅读，不得不说，谷歌爸爸的论文干货真的很多，有很多细节的东西本文都没写进去，读完之后也是受益匪浅。嗯，好好学习！

# 参考文献

[1] MARTINEC, E., 2008. Noise, dynamic range and bit depth in digital SLRs, http://theory.uchicago.edu/∼ejm/pix/20d/tests/noise.

[2] HASINOFF, S. W., DURAND, F., AND FREEMAN, W. T. 2010. Noise-optimal capture for high dynamic range photography. CVPR.

[3] ZHANG, L., DESHPANDE, A., AND CHEN, X. 2010. Denoising vs. deblurring: HDR imaging techniques using moving cameras. CVPR.

[4] JOSHI, N., AND COHEN, M. F. 2010. Seeing Mt. Rainier: Lucky imaging for multi-image denoising, sharpening, and haze removal. ICCP.

[5] HEALEY, G., AND KONDEPUDY, R. 1994. Radiometric CCD camera calibration and noise estimation. TPAMI 16, 3, 267–276.