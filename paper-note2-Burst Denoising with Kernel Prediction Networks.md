# 【paper-note2】Burst Denoising with Kernel Prediction Networks

> 论文作者：Ben Mildenhall Google Research & UC Berkeley
>
> 收录：CVPR2018

## 核心思想

本文针对连续图像序列降噪做出改进，运用了KPN（我觉得就是Unet的变种）进行降噪核预测，对图像序列中的每一张图片都预测一个滤波器，将滤波器作用于图像序列的每张图上，最后对整个图像序列取平均得到输出$\hat{Y}$ ，以此达到对齐和降噪的目的。



## 参考文献

[1] G. Healey and R. Kondepudy. Radiometric CCD camera calibration and noise estimation. TPAMI, 1994.

## 补充材料

![](https://raw.githubusercontent.com/Badstu/pic_set/master/img/20190721164328.png)