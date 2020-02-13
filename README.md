# DeepCalib
DeepCalib: A Deep Learning Approach for Automatic Intrinsic Calibration of Wide Field-of-View Cameras http://cml.kaist.ac.kr/projects/DeepCalib

## Table of contents

- [Requirements](#requirements)
- [Dataset generation](#dataset-generation)
- [Training DeepCalib](#training-deepcalib)
- [Notes](#notes)
  - [Different models](#different-models)
  - [Smth](#smth)
- [Citation](#citation)

## Requirements
- Python 2.7
- Keras 2.1
- TensorFlow 1.4

## Dataset generation
We provided the code for the whole data generation pipeline. First you have to download sun360 dataset using [a link](https://github.com/alexvbogdan/DeepCalib/blob/master/dataset/download_images.py). Then, you have to choose whether your dataset is gonna have continuous or discrete values. We provide the dataset generation code for both in [a link](https://github.com/alexvbogdan/DeepCalib/blob/master/dataset/).

## Training DeepCalib

## Notes

#### Different models

#### Smth

## Citation

@inproceedings{bogdan2018deepcalib,
  title={DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras},
  author={Bogdan, Oleksandr and Eckstein, Viktor and Rameau, Francois and Bazin, Jean-Charles},
  booktitle={Proceedings of the 15th ACM SIGGRAPH European Conference on Visual Media Production},
  pages={1--10},
  year={2018}
}
