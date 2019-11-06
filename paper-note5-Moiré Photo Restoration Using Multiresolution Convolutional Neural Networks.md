# 【paper-note5】Moiré Photo Restoration Using Multiresolution Convolutional Neural Networks

> 论文题目：《Moiré Photo Restoration Using Multiresolution Convolutional Neural Networks》
>
> 论文作者：Yujing Sun The University of Hong Kong
>
> 收录：TIP 2018

## 前言

最近调研了一下去马赛克和去摩尔纹的一些工作，并开始逐渐明确自己的研究方向。

本论文是香港大学sun同学的工作，这位同学以前的工作有两个，分别是基于L0最小化的图像结构检索[1]（A）和基于L0最小化的点云降噪[2]（B），再加上这篇也是A类期刊，感叹一下大佬就是大佬，两篇A一篇B，要是我能达到这种境界也算是不留遗憾了吧。

去摩尔纹的任务和其他复原任务不同，摩尔纹所涉及的频域带比较宽，从低频到高频都有，另外，任务要求将图片尽可能复原成原样，所以用PSNR有较高置信度。

## 核心思想

本文核心思想是利用一个多分辨率全卷积网络对`摄屏图像`的摩尔纹去除，其实想法很简单，也是端到端的神经网络，损失函数是像素级的L2距离。多分辨率网络和HR-net很像，不过下采样是全卷积网络，上采样是膨胀卷积，输出是简单相加，算是比较简单方法：

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191105223517.png)

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191105223611.png)

之所以能发A，有如下几点：

1. 据作者自己说，他是第一个做摄屏图像去摩尔纹的，既然它在顶刊上，就姑且当他是真的吧。
2. 创造了个摩尔纹数据集，获取过程比较有意思，这真的是要靠丰富的图像处理经验了，稍后会提。不过他没公开数据集，复现的话，要费一番力气。
3. 图像对齐算法比较有意思，用PSNR判定是否对齐。

## 主要工作

和一般性的摩尔纹图像不同，本文针对的是相机拍摄显示屏的时候，由于显示屏的显示像素阵列和相机传感器的感光像素阵列进行重叠，发生了差拍现象，如下图

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191106112627.png)

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191106105417.png)

### 网络

提出了一个新的多分辨率全卷积网络，网络结构如图，其中下采样和上采样都是用了卷积核的方式，最后是简单的sum，得到输出，loss用了像素级的L2距离。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191106105543.png)

之所以要用多分辨率，作者说去摩尔纹的问题在于如何移除多频率的artifacts，而其他图像复原问题包括去马赛克的问题在于如何去掉或者恢复高频信息，多分辨率的主要依据是图像缩小后，上一个level的一部分高频就会变成下一个level的低频，每个分支处理一个或多个频率，最后融合到一起。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191106110803.png)

此多分辨率和图像金字塔最大的差别就是前者是非线性的（卷积+RELU），后者是线性的（下采样）。

### 数据集

网络定义好了之后最重要的就是数据集了，摄屏图像去摩尔纹没有人做过，所以数据集要自己收集，作者从`ImageNet ISVRC 2012 dataset`获取了135,000张图片对，分别包含了摩尔纹图片和groud truth，patch大小为256\*256，90%训练集，10%测试集。

* 采集方法

  用了三个手机：iphone 6, 三星s7 edge, 索尼 Xperia Z5 Premium Dual，和三个显示屏：macbook pro retina（2560X1600），Dell U2410 LCD（1920X1200），Dell SE198WFP LCD（1280X800），两两组合的方式共有9种组合。

  对于每种组合，从ImageNet种抽取15,000张图片对，做法是把每张图片在白色的显示器上面轮播，每张图片0.3秒（1/3），然后用手机对着显示屏**拍视频**（骚操作，在看到这个之前我还以为是一张一张拍的。。），因为24帧/秒，所以视频按0.3秒的间隔抽取关键帧就可以了。

* 对齐方法

  因为手机拍照时视角会有倾斜，而ImageNet是平面的，好在显示屏也是平面的，理论上只要有对应的特征点然后做单应性变换就可以拉回来，具体做法如下。

  ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191106112607.png)

  对原始图像加黑边，并加四个黑色矩形，二值化，按顺序标好角点（b中绿色的点），对获取的摩尔纹图像二值化，然后用Harris 角点检测方法检测角点（e中红色的点），检测出来的点会多于标定点，如下图

  ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191106113212.png)

  这时候检测角点11X11领域内的黑色像素和白色像素之比，如果接近3或1/3，则认为角点存在，否则抹去；另外规定最近邻角点的距离阈值，小于这个阈值就只选择其中一个。

  得到了标定的20个点和检测的20个点后，就可以做3X3的单应性变换（homography），至此对齐就完成了。

## 其他方面

论文还做了一些其他的工作，增加工作量，包括了如下几点：

* 对灰度图像做验证，拿了近似灰度图的RGB图像和真实灰度图送到网络里，发现效果差不多。这是验证了网络真的学习到了摩尔纹，而不是简单的色彩伪影。

* 拿了超分的VDSR，降噪的DnCNN，IRCNN，纹理去除的RTV和SDF，去模糊的pyramidCNN和IRCNN，以及图像分割的U-Net做对比，并给出了训练过程。

  ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191106115152.png)

* 对自己的网络改了一些配置，横向比较。

  ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191106115332.png)

* 用户研究

  * 作者居然还做了用户研究，这在计算机论文里面不常见，找了60个人发问卷，20道题，每道题六张图片，分别是VDSR，DnCNN，PyramidCNN，U-net，V_Concate和上面的方法，让调查者选1到2张最好看的图片，结果如下。

    ![1573012998641](C:\Users\zqws1\AppData\Roaming\Typora\typora-user-images\1573012998641.png)

* 探索了模型泛化性

  * 用华为P9拍的照片做测试
  * 在正常图像中嵌入一部分摩尔纹图像测试
  * 用非摄屏摩尔纹图像测试

* Limitation和future work

  * 在提到limitation时，除了一些不可解释的样例，作者说该方法会带来图像的模糊，原因可能是相机移动，对齐失败（可以找更好的对齐方法），原始高频信息被摩尔纹破坏，以及20个标定点可能会被摩尔纹破坏从而导致检测失误。

  ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191106115820.png)

  * future work有三点：
    * 对摩尔纹分类
    * 更好摩尔纹描述方法（客观，主观）
    * 扩充数据集，在更多环境下拍摄。



## 可以改进的点

1. 用简化版的HR-net代替多分辨率网络。

2. 引入残差的思想，让网络去学习其残差。

3. GAN-based demosaic[3]思想可以用在去摩尔纹上，前面的网络作为一个generate，在训练一个discriminate来作为质量评估，做法就是输入restore image和GT，训练D网络，这点在下一篇paper-note会提到。另外引入其他的loss（perceptive loss和discriminate loss）可能也能提高精度。

   ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191105225836.png)

4. 能否用马赛克bayer图像替代RGB图像作为输入，或者借鉴CFA-design[4]的思想，生成一个新的去马赛克方法，真正的端到端。



## 参考文献

[1] Sun, Yujing & Schaefer, Scott & Wang, Wenping. (2017). Image Structure Retrieval via L0 Minimization. IEEE Transactions on Visualization and Computer Graphics. PP. 1-1. 10.1109/TVCG.2017.2711614. 

[2] Sun, Yujing & Schaefer, Scott & Wang, Wenping. (2015). Denoising point sets via L 0 minimization. Computer Aided Geometric Design. 35-36. 10.1016/j.cagd.2015.03.011. 

[3] Dong W , Yuan M , Li X , et al. Joint Demosaicing and Denoising with Perceptual Optimization on a Generative Adversarial Network[J]. Journal of Taiyuan University of Science and Technology, 2018.

