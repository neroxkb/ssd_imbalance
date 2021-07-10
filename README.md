# 模型SSD
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/1.png)<br>
SSD算法中使用到了conv4_3,conv_7，conv8_2,conv7_2,conv8_2,conv9_2,conv10_2,conv11_2这些大小不同的feature maps，其目的是为了能够准确的检测到不同尺度的物体，因为在低层的feature map,感受野比较小，高层的感受野比较大，在不同的feature map进行卷积，可以达到多尺度的目的。
<br>
SSD中的Defalut box和Faster-rcnn中的anchor机制很相似。就是预设一些目标预选框，后续通过softmax分类+bounding box regression获得真实目标的位置。对于不同尺度的feature map 上使用不同的Default boxes。
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/2.png)<br>
选取的feature map包括38x38x512、19x19x1024、10x10x512、5x5x256、3x3x256、1x1x256，Conv4_3之后的feature map默认的box是4个，在38x38的这个平面上的每一点上面获得4个box，那么我们总共可以获得38x38x4=5776个；同理，依次将FC7、Conv8_2、Conv9_2、Conv10_2和Conv11_2的box数量设置为6、6、6、4、4，那么可以获得的box分别为2166、600、150、36、4，即总共可以获得8732个box，然后将这些box送入NMS模块中，获得最终的检测结果。

# 数据集
训练时将数据集转换成了VOC格式，总共5500张图像，其中3500+3500=7000作为训练集，1500+1500=3000作为测试集
<br>

# 样本不均衡优化方法
1.设置采样权重，使用WeightRandomSampler函数，带电芯充电宝的样本数量少，给它设置的权重是不带电芯充电宝的10倍
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/3.png)
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/4.png)<br>

设置采样权重之后，每次训练时，获取的类别数量，class0是背景，每个class设置8732个框，两个类别就是17464，所以class0+1+2总量为17464；class1是带电芯充电宝；class2是不带电芯充电宝，可以看出class1的数量有明显的提升，相对于没有采样权重的情况，设置采样权重可以对样本量少的类别进行更多的训练。
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/5.png)
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/6.png)<br>

2.数据增强<br>
通过数据增强增加样本数量，包括平移、旋转、裁剪、镜像等
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/7.png)
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/8.png)<br>

# 训练与结果
1.（1）设置配置文件，包括训练次数，lr的修改，features_maps等参数
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/9.png)<br>
（2）修改voc文件
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/10.png)<br>
（3）建立网络
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/11.png)<br>
（4）Loss值
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/12.png)
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/13.png)
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/14.png)<br>


2.评估函数<br>
加载网络，并对测试集进行测试
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/15.png)
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/16.png)<br>
3.结果对比<br>
上面为不使用采样器的结果，下面为使用采样器的结果，结果有较小的提升，由于样本数量较少，即使多次采样，还是那500张的重复训练，效果比不上5000样本数量的类别
<br>
样本未增强结果0.592
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/17.png)<br>
<br>
样本增强的结果0.783
<br>
![](https://github.com/neroxkb/ssd_imbalance/raw/master/doc/18.png)<br>























