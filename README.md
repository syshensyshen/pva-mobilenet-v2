# pva-mobilenet-v2

### Introduction

This is a Caffe implementation of Google's MobileNets (v1 and v2). For details, please read the following papers:
- [v1] [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [v2] [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)

try using pvanet architecture to trained voc data

### Pretrained Models on ImageNet using https://github.com/shicai/MobileNet-Caffe.git

Network|Top-1|Top-5|sha256sum|Architecture
:---:|:---:|:---:|:---:|:---:
MobileNet v1| 70.81| 89.85| 8d6edcd3 (16.2 MB) | [netscope](http://ethereon.github.io/netscope/#/gist/2883d142ae486d4237e50f392f32994e)
MobileNet v2| 71.90| 90.49| a3124ce7 (13.5 MB)| [netscope](http://ethereon.github.io/netscope/#/gist/d01b5b8783b4582a42fe07bd46243986)


#### accuary

AP for aeroplane = 0.5731
AP for bicycle = 0.6798
AP for bird = 0.4748
AP for boat = 0.4037
AP for bottle = 0.3515
AP for bus = 0.7517
AP for car = 0.7366
AP for cat = 0.7646
AP for chair = 0.2972
AP for cow = 0.3399
AP for diningtable = 0.5541
AP for dog = 0.7086
AP for horse = 0.7416
AP for motorbike = 0.7236
AP for person = 0.7001
AP for pottedplant = 0.2407
AP for sheep = 0.0702
AP for sofa = 0.5284
AP for train = 0.6882
AP for tvmonitor = 0.4034
Mean AP = 0.5366


#pva-mobilenet-v2 model:
url: https://pan.baidu.com/s/1Cl4MXiU7otkB6bxGIOLxwg
