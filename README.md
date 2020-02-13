# DeepCalib
DeepCalib: A Deep Learning Approach for Automatic Intrinsic Calibration of Wide Field-of-View Cameras http://cml.kaist.ac.kr/projects/DeepCalib

## Table of contents

- [Short description](#short-description)
- [Requirements](#requirements)
- [Dataset generation](#dataset-generation)
- [Training DeepCalib](#training-deepcalib)
- [Notes](#notes)
  - [Different architectures](#different-architectures)
  - [Weights](#weights)
- [Citation](#citation)

## Short description
We present a novel fully automatic deep learning-based approach works with a single image of general scenes. Our approach builds upon Inception-v3 architecture: our network **automatically estimates the intrinsic parameters of the camera** (focal length and distortion parameter) from a **general single input image**.

## Requirements
- Python 2.7
- Keras 2.1
- TensorFlow 1.4

## Dataset generation
We provided the code for the whole data generation pipeline. First you have to download sun360 dataset using [download script](https://github.com/alexvbogdan/DeepCalib/blob/master/dataset/download_images.py). Then, you have to choose whether your dataset is going to have continuous or discrete values. We provide the dataset generation code for both in a dataset [folder](https://github.com/alexvbogdan/DeepCalib/blob/master/dataset/).

## Training DeepCalib
To train DeepCalib you need to choose which architecture you want to use (refer to the `Section 3.3` of [our paper](https://drive.google.com/file/d/1pZgR3wNS6Mvb87W0ixOHmEVV6tcI8d50/view)). This repo contains all the training scripts for both classification and regression networks as well as all 3 architectures mentioned in the paper.

## Notes

#### Different architectures
For detailed information refer to the `Section 4.2` of [our paper](https://drive.google.com/file/d/1pZgR3wNS6Mvb87W0ixOHmEVV6tcI8d50/view). In short, `SingleNet` is the best network for predicting focal length and distortion parameter in terms of accuracy. In addition, since it is a single network contrary to `DualNet` and `Seqnet` it is computationally cheaper to use the former.

#### Weights
The weights for our network can be found [here](https://drive.google.com/file/d/1TYZn-f2z7O0hp_IZnNfZ06ExgU9ii70T/view). We recommend to use `SingleNet` since we experimentally confirmed it outperforms the other ones. The regression weights for `DualNet` are currently wrong, although you can train your own.

## Citation
```
@inproceedings{bogdan2018deepcalib,
  title={DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras},
  author={Bogdan, Oleksandr and Eckstein, Viktor and Rameau, Francois and Bazin, Jean-Charles},
  booktitle={Proceedings of the 15th ACM SIGGRAPH European Conference on Visual Media Production},
  year={2018}
}
```
