# IncepTNet
A Large-Scale Drone based Thermal Infrared Benchmark and Inception Transformer Network for Crowd Counting([paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320325014414)).

## Abstract
Crowd counting plays a crucial role in applications like public safety and smart cities, where estimating the number of people in images or videos is essential. However, since most of the current crowd counting datasets are visible light images, the counting performance is limited by the light intensity. To tackle this problem, we collect a drone based thermal infrared crowd counting dataset named LYU-DroneInfrared, which includes 64,210 images and 2,997,352 head annotation points, covering different scenes such as schools, streets, squares, sports grounds, etc. In addition, we propose the IncepTNet, an inception transformer network based on the transformer architecture. It consists of two parts: low frequency feature extraction and high frequency feature extraction. In low-frequency feature extraction, average pooling is first applied, followed by multi-head self-attention to capture contextual information in the image. On the other hand, to be able to pay more attention to the fine-grained details of the images, the parallel approach of convolution and maximum pooling are used to extract the high frequency features. The proposed method is validated on two benchmark datasets JHU-Crowd++ and NWPU-Crowd, and a newly collected LYU-DroneInfrared dataset. Extensive experimental results have shown that the IncepTNet method exhibits excellent crowd counting performance on different types of datasets.

##  Dataset
The LYU-DroneInfrared dataset was captured by drone equipped with an infrared camera. It includes 64,210 frames from 237 sequences, covering the playgrounds, streets, squares, basketball courts and other scenes. The dataset contains a total of 2,997,352 annotation points and 11,612 people, and each image has a resolution of 640 × 512. We partitioned the dataset into training, validation, and test sets following a ratio of 6:1:3. The training set comprised 39,289 images, the validation set consisted of 5,841 images, and the test set contained 18,980 images.

<img src="/fig/fig1.png" width="600" alt="LYU-DroneInfrared">

##  Demo:
<img src="/fig/demo.gif" width="600" alt="Demo">

##  BaiduYun:
[LYU-DroneInfrared](https://pan.baidu.com/s/1_0mnXHiqsscmhGxPsKlutQ?pwd=y4u5) (code:y4u5)

##  Code:
[Code](https://github.com/TIMOLEEGO/IncepTNet)

##  Citation:
Please cite this paper if you want to use it in your work.

```bibtex
@article{wang2025large,
  title={A large-scale drone based thermal infrared benchmark and inception transformer network for crowd counting},
  author={Wang, Xing and Li, Timing and Liu, Ya and Yao, Shuanglong and Liu, Ye and Yang, Nan and Zhu, Pengfei},
  journal={Pattern Recognition},
  pages={112778},
  year={2025},
  publisher={Elsevier}
}
