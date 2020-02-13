# DeepCalib
The implementation of our [2018 CVMP DeepCalib](https://drive.google.com/file/d/1pZgR3wNS6Mvb87W0ixOHmEVV6tcI8d50/view) paper. 

## Table of contents

- [Short description](#short-description)
- [Requirements](#requirements)
- [Dataset generation](#dataset-generation)
- [Training DeepCalib](#training-deepcalib)
- [Camera calibraition](#camera-calibration)
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
We provided the code for the whole data generation pipeline. First you have to download sun360 dataset using this [download script](https://github.com/alexvbogdan/DeepCalib/blob/master/dataset/download_images.py). Then, you have to choose whether your dataset is going to have continuous or discrete values. We provide the dataset generation code for both in a dataset [folder](https://github.com/alexvbogdan/DeepCalib/blob/master/dataset/).

## Training DeepCalib
To train DeepCalib you need to choose which architecture you want to use (refer to the `Section 3.3` of [our paper](https://drive.google.com/file/d/1pZgR3wNS6Mvb87W0ixOHmEVV6tcI8d50/view)). This repo contains all the training scripts for both classification and regression networks as well as all 3 architectures mentioned in the paper. In both regression and classification [Seq_Net](https://github.com/alexvbogdan/DeepCalib/tree/master/network_training/Classification/Seq_Net) and [Dual_Net](https://github.com/alexvbogdan/DeepCalib/tree/master/network_training/Classification/Dual_Net) folders "dist" and "focal" refer to the netwotks used for distortion parameter and focal length, respectively. All the training codes are available in this [folder](https://github.com/alexvbogdan/DeepCalib/tree/master/network_training).

## Camera Calibration
To infer distortion parameter and focal length of a given camera we take a short video, extract the frames and run the prediction on all of them. After that, we take the mean or the median of predicted values and use that as a final result. However, in a slight modification you can use them for a single image prediction as well. Below you can see some of the results of image rectification using parameters obtained from single image calibration. ![Results](https://github.com/alexvbogdan/DeepCalib/blob/master/Results.png)
In [prediction folder](https://github.com/alexvbogdan/DeepCalib/tree/master/prediction) we have the codes for all the networks except for `SeqNet` regression because the weights for this architecture are currently unavailable. We uploaded a simple python script for frame extraction from video sequence.

## Notes

#### Different architectures
For detailed information refer to the `Section 4.2` of [our paper](https://drive.google.com/file/d/1pZgR3wNS6Mvb87W0ixOHmEVV6tcI8d50/view). In short, `SingleNet` is the best network for predicting focal length and distortion parameter in terms of accuracy. In addition, since it is a single network contrary to `DualNet` and `Seqnet`, it is computationally cheaper to use the former. ![DeepCalib architectures](https://github.com/alexvbogdan/DeepCalib/blob/master/DeepCalib_architectures.png)

#### Weights
The weights for our network can be found [here](https://drive.google.com/file/d/1TYZn-f2z7O0hp_IZnNfZ06ExgU9ii70T/view). We recommend to use `SingleNet` since we experimentally confirmed it outperforms the other ones. The regression weights for `SeqNet` are currently unavailable, although you can train your own.

## Citation
```
@inproceedings{bogdan2018deepcalib,
  title={DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras},
  author={Bogdan, Oleksandr and Eckstein, Viktor and Rameau, Francois and Bazin, Jean-Charles},
  booktitle={Proceedings of the 15th ACM SIGGRAPH European Conference on Visual Media Production},
  year={2018}
}

@inproceedings{xiao2012recognizing,
  title={Recognizing scene viewpoint using panoramic place representation},
  author={Xiao, Jianxiong and Ehinger, Krista A and Oliva, Aude and Torralba, Antonio},
  booktitle={2012 IEEE Conference on Computer Vision and Pattern Recognition},
  year={2012},
}
```
