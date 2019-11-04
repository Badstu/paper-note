# 【paper-note4】Reconﬁguring the Imaging Pipeline for Computer Vision

> 论文题目：《Reconﬁguring the Imaging Pipeline for Computer Vision》
>
> 论文作者：Mark Buckler Cornell University
>
> 收录：ICCV2017

## 前言

ISP在前面几篇论文笔记里面已经多次介绍过了，这篇论文研究点比较新奇。针对常见的计算机视觉任务对现有ISP进行改进，发现传统摄像ISP中对视觉任务影响最大的也只有**去马赛克**和**gamma矫正**。其主要目的是在不大幅损失精度的条件下提升效率，节约能源（save energy），据作者自己说，相对于正常的photography mode，其提出的vision mode 能够节约平均75%的能源。

### 论文中有意思的点如下：

1. 把传统ISP直接弃用，用low power vision mode产生低质量的图片，这个图片只适用于视觉任务。这个骚操作只了损失一点点精度，但是大大节省了能源。

2. 用了八个计算机视觉算法，包含五个神经网络方法，三个传统方法，验证vision mode的精度和效率，想法可行（其实这是废话，论文的experiment再不行都要说行。）

3. 基于Kim[1]提出的Pipline 反向操作，把RGB图片反转为RAW格式文件，然后再模拟自己的ISP，输出新的RGB图像，用Halide写的，实现发布在github上：https://github.com/cucapra/approx-vision，他叫CRIP（Conﬁgurable & Reversible Imaging Pipeline），和上一篇论文的unprocess想法有点像，不过是别人的想法，他把他实现了一下。

   这个操作的目的是，直接获得raw文件比较麻烦，有工具能生成raw文件，好做研究。

4. 消融研究，大概意思就是控制变量法，看看每一个ISP stage对视觉效果的影响有多大。

### 我自己的一些想法

1. 结合去马赛克等图像复原的工作，比起那些论文直接下采样生成RAW文件，反向ISP的做法好很多了。问题也很明显，反向ISP，那复原的时候就不能只做去马赛克，而且用PSNR作为指标明显就不是很好了。这点需要思考一下。
2. 神经网络虽然很厉害，但是现在神经网络的输入最多还只是几百*几百像素的图片，摄影领域上千万级的像素没有充分利用起来，5G时代的4K咋办，虽然能做超分，但是高级视觉任务还是受限于算力，如何把高分辨率的高清图片用作视觉任务，这篇文章也是给出了一个新思路。

## 论文工作

### Vision mode pipline

首先，论文审视了现在标准的ISP并且提出了一个适用于**计算机视觉任务**的ISP。在信号处理的初始阶段，简单的完成去马赛克和gamma校正以及色调映射相应功能，在放大器中运用Power Gated操作，控制图片分辨率；用下采样操作模拟去马赛克；在ADC模数转换中运用Logarithmic函数处理gamma校正。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0JhZHN0dS9waWNfc2V0L21hc3Rlci9pbWcvMjAxOTExMDQxNDQ4NTcucG5n?x-oss-process=image/format,png)

### CRIP

提出了一个**可配置的反转ISP**（CRIP，Configurable & Reversible Imaging Pipeline.
pression.），有点像图像重上色和校正白平衡的操作。算法用Halide写的，发布到了github上：https://github.com/cucapra/approx-vision

​		![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191104150649.png)

- 基于Kim[1]提出reversible ISP model，把RGB图片反转为RAW格式文件，然后再模拟自己的ISP（vision mode），输出新的RGB图像；
- 在这个过程中，除了上述的model，还利用了两个方法对去噪和去马赛克进行增强，RGB去噪图片已经没有噪声了，为了保留其噪声信息，用了Chehdi[2]提出的传感器噪声模型回复噪声；
- 反转去马赛克的操作依旧是按照bayer格式移除另外两个channel的值。
- CRIP也有个缺点，就是真实的raw文件时12位的，而反转后的raw文件时8位的，这也是改进点。

### 消融实验

选取了八种计算机视觉算法，包含5种深度学习算法和3种传统算法，涵盖了分类，目标检测，人脸检测，光流分析，结构恢复五种任务。

作者提出ISP的通用模式：**1. 去马赛克， 2. 去噪，3. 色彩转换，4. 色调映射，5. gamma校正**，针对每一个ISP步骤，配置CRIP并应用到8种算法上，观察精度损失大小。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191104151638.png)

探索所有步骤的组合太多了，所以作者把配置分为两种，一种是禁用其中一个stage，看看前后精度的对比；另一种是只启用其中一个stage，看看和正常的ISP精度对比。

结论如下图，黑线代表正常ISP的精度，虚线代表禁用所有ISP的精度，从图（a）可以观察到，*色彩转换*和*色域映射*对精度影响是最低的，禁用后的效果很接近黑线，基本没什么精度损失，从图（b）也可以观察到，启用*色彩转换*和*色域映射*也并不会带来太多的error下降。

但是对于去马赛克和gamma校正来说恰恰相反，禁用这两个stage和启用这两个stage带来的精度变化都很大，其次是去噪，大部分算法对去噪不敏感，但是SGBM算法是对噪声敏感的，因此，去噪也能带来很大的精度变化。另外OpenFace在只开启去马赛克和gamma校正的时候比baseline的效果还要好，论文作者说这是由于算法的随机性导致的，因为在跑了十次训练之后，发现这个算法的error在8.22%到10.35%之间以标准差为0.57%进行震荡。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191104152627.png)

基于以上实验结果，提出了只包含去马赛克和gamma校正的最小化ISP，做了实验，并且可视化归一化的error。结论和上面相同，只采用去马赛克和gamma校正这两个步骤并不会对结果造成太大损失，当加入去噪后，损失会变得更小。

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191104164248.png)

### 传感器重新设计

找到了ISP中影响最大的两个stage之后，这一步就是对这两个stage重新设计，用硬件操作代替软件算法。作者提出了如下三个操作。

1. `adjustable resolution via selective pixel readout and power gating`，这一步是对分辨率做一些改动，正如上面思考的一样，目前的计算机视觉任务没有发挥高清图片的作用，输入低分辨率的图片就能满足算法所需。

   传统ISP是通过列并行融合的方式进行读取数据，每一列的像素都会被送入放大器和ADC，本文提出的**power-gating column amplifiers and ADC**能够有选择性的读数，从而读出ROI(region of interest)，降低分辨率。

   具体实现如下图，我没学过数字电路，大概意思就是用power gating作为一个选择器，以此读出ROI并送入ADC中。

   ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191104164308.png)

2. `subsampling to approximate ISP-based demosaicing`，这一步是模拟去马赛克的过程，高级的插值算法是为了得到精细高质量的图片，但是计算机视觉不需要这么精细的图片，比如说现在的手机都能拍4800万像素了，实际上视觉算法能处理800*800=64万像素就已经很不错了，很直观的一种思想是直接做下采样，即把4800万下采样到64万。如下图，论文作者把bayer图像中的每个2\*2小块下采样成1个像素点，把两个绿色像素丢掉一个，把剩余的一通道RGB像素点组成一个三通道的彩色像素点。

   ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191104164333.png)

   众所周知，ISP中耗时和消耗资源最多的是去马赛克算法，简单下采样操作能极大减少算法的时间复杂度。（这点可以为以后的科研提供idea。）

3. `nonlinear ADC quantization to perform gamma compression`，自然光强的概率分布函数是对数分布的，而传统模数转化的量化操作是均匀分布的，因此作者提出用logarithmic量化ADC代替原来的ADC，非线性量化能使用更少的位数储存。通常是gamma校正来恢复光强对数分布，因此该操作相当于模拟了gamma校正。（这里有个疑问，raw格式的优点是值和光强是线性关系，如果进行非线性量化得到的值应该不是线性相关的值了。如何处理？能否轻易转化为线性的值。）

   ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191104164358.png)

### 实验

当然作者虽然说硬件重新设计，但是实验还是软件做的。

1. 去马赛克，用了CIFAR-10，将下采样和双线性、nn以及真实去马赛克图像做比较，发现下采样的error损失不大。

   ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191104165429.png)

2. 非线性量化时用对数量化代替线性量化，下图展示了两种量化关系，量化位数和error的关系，可以看到对数量化对更少的位数有更好的容错性。最后是采用了5位对数ADC代替12位线性ADC。

   ![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20191104164217.png)

3. 分辨率更改，文章中的做法是从ImageNet中选择了15000张包含了CIFAR-10的十个类别的高分辨率照片，先下采样1/4，1/16和1/64，再把缩小后的图片用OpenCV edge-aware scaling缩放到32\*32大小，对照组的做法是不进行下采样，直接缩放到32\*32大小，结果表明，下采样并没有带来太大的损失。

   | 方法     | 没有下采样的error | 下采样后的error |
   | -------- | ----------------- | --------------- |
   | LeNet    | 39.6%             | 40.9%           |
   | ResNet20 | 26.34%            | 27.71%          |
   | ResNet44 | 24.50%            | 26.5%           |

   

### 开销衡量

论文最后一部分是衡量每个模块的能源开销。

1. 传感器ADC，ADC读出占了传感器总功耗的**50%**，采用5位对数ADC能够比12位线性ADC节省**99.95%**的能源消耗，所以采用logarithmic ADC能节省将近一半的耗能。
2. 分辨率，根据文献[3]，传感器读出，I/O，和像素阵列，总共占了95%的能耗，而这些操作和分辨率大小是线性相关的，因此减少分辨率能相应减少能耗。
3. ISP，用了电路设计就能绕开ISP，所以这方面的能耗就没有了。
4. 总功耗，image pipline包含了两方面的能耗，传感器和ISP，由于视觉模式下的ISP被完全禁用，再加上用了5-bit 对数ADC，总能耗能够节省将近75%。当分辨率进一步降低，能耗能够更低。
5. 当然上面的能耗分析没有考虑到，power gating，额外的多路复用技术和芯片外的交互，这些都是future work了。

### 总结

本文用一个vision mode的ISP代替了传统的ISP，用实验针对具体的算法验证了其效果，但是没有给出理论分析和统计建模，作者说这是follow-on可以做的；

另外硬件设计在本文中只是提了一下，并没有完整的设计；在做能耗评估的时候给出了第一直观印象的能耗，没有考虑到硬件管理和布局的能耗；

最后，本文其实提供了一个很好的思路，就是在设计**计算机视觉算法**的时候要和**相机系统**进行协同设计，结合软硬件才是未来移动设备的大趋势！

## 参考文献

[1] S. J. Kim, H. T. Lin, Z. Lu, S. S¨usstrunk, S. Lin, and M. S. Brown. A new in-camera imaging model for color computer vision and its application. IEEE Transactions on Pattern Analysis and Machine Intelligence, 34(12):2289–2302, Dec. 2012.

[2] M. L. Uss, B. Vozel, V. V. Lukin, and K. Chehdi. Image informative maps for component-wise esti- mating parameters of signal-dependent noise. Jour- nal of Electronic Imaging, 22(1):013019–013019, 2013.

[3] Y. Chae, J. Cheon, S. Lim, M. Kwon, K. Yoo, W. Jung, D. H. Lee, S. Ham, and G. Han. A 2.1 M pixels, 120 frame/s CMOS image sensor with column-parallel δσ ADC architecture. IEEE Journal of Solid-State Circuits, 46(1):236–247, Jan. 2011