# Video Swin Transformer

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-swin-transformer/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=video-swin-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-swin-transformer/action-classification-on-kinetics-600)](https://paperswithcode.com/sota/action-classification-on-kinetics-600?p=video-swin-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-swin-transformer/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=video-swin-transformer)

By [Ze Liu](https://github.com/zeliu98/)\*, [Jia Ning](https://github.com/hust-nj)\*, [Yue Cao](http://yue-cao.me),  [Yixuan Wei](https://github.com/weiyx16), [Zheng Zhang](https://stupidzz.github.io/), [Stephen Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en) and [Han Hu](https://ancientmooner.github.io/).

This repo is the official implementation of ["Video Swin Transformer"](https://arxiv.org/abs/2106.13230). It is based on [mmaction2](https://github.com/open-mmlab/mmaction2).

## Updates

***06/25/2021*** Initial commits

## Introduction

**Video Swin Transformer** is initially described in ["Video Swin Transformer"](https://arxiv.org/abs/2106.13230), which advocates an inductive bias of locality in video Transformers, leading to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization. The locality of the proposed video architecture is realized by adapting the Swin Transformer designed for the image domain, while continuing to leverage the power of pre-trained image models. Our approach achieves state-of-the-art accuracy on a broad range of video recognition benchmarks, including action recognition (`84.9` top-1 accuracy on Kinetics-400 and `86.1` top-1 accuracy on Kinetics-600 with `~20x` less pre-training data and `~3x` smaller model size) and temporal modeling (`69.6` top-1 accuracy on Something-Something v2).


![teaser](figures/teaser.png)

## Results and Models

### Kinetics 400

| Backbone |  Pretrain   | Lr Schd | spatial crop | acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Swin-T  | ImageNet-1K |  30ep   |     224      |  78.8  |  93.6  |   28M   |  87.9G  |  [config](configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py)  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth)/[baidu](https://pan.baidu.com/s/1mIqRzk8RILeRsP2KB5T6fg) |
|  Swin-S  | ImageNet-1K |  30ep   |     224      |  80.6  |  94.5  |   50M   |  165.9G  |  [config](configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py)   | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth)/[baidu](https://pan.baidu.com/s/1imq7LFNtSu3VkcRjd04D4Q) |
|  Swin-B  | ImageNet-1K |  30ep   |     224      |  80.6  |  94.6  |   88M   |  281.6G  |  [config](configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py)   | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth)/[baidu](https://pan.baidu.com/s/1bD2lxGxqIV7xECr1n2slng) |
|  Swin-B  | ImageNet-22K |  30ep   |     224      |  82.7  |  95.5  |   88M   |  281.6G  |  [config](configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py)   | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth)/[baidu](https://pan.baidu.com/s/1CcCNzJAIud4niNPcREbDbQ) |

### Kinetics 600

| Backbone |  Pretrain   | Lr Schd | spatial crop | acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Swin-B  | ImageNet-22K |  30ep   |     224      |  84.0  |  96.5  |   88M   |  281.6G  |  [config](configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py)   | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth)/[baidu](https://pan.baidu.com/s/1ZMeW6ylELTje-o3MiaZ-MQ) |

### Something-Something V2

| Backbone |  Pretrain   | Lr Schd | spatial crop | acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Swin-B  | Kinetics 400 |  60ep  |     224      |  69.6  |  92.7  |   89M   |  320.6G  |  [config](configs/recognition/swin/swin_base_patch244_window1677_sthv2.py)   | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth)/[baidu](https://pan.baidu.com/s/18MOGf6L3LeUjrLoQEeA52Q) |

**Notes**:

- **Pre-trained image models can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer)**.
- The pre-trained model of SSv2 could be downloaded at [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_kinetics400_22k.pth)/[baidu](https://pan.baidu.com/s/1ZnJuX7-x2BflDKHpuvdLUg).
- Access code for baidu is `swin`.

## Usage

###  Installation

Please refer to [install.md](docs/install.md) for installation.

We also provide docker file [cuda10.1](docker/docker_10.1) ([image url](https://hub.docker.com/layers/ninja0/mmdet/pytorch1.7.1-py37-cuda10.1-openmpi-mmcv1.3.3-apex-timm/images/sha256-06d745934cb255e7fdf4fa55c47b192c81107414dfb3d0bc87481ace50faf90b?context=repo)) and [cuda11.0](docker/docker_11.0) ([image url](https://hub.docker.com/layers/ninja0/mmdet/pytorch1.7.1-py37-cuda11.0-openmpi-mmcv1.3.3-apex-timm/images/sha256-79ec3ec5796ca154a66d85c50af5fa870fcbc48357c35ee8b612519512f92828?context=repo)) for convenient usage.

###  Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.
The supported datasets are listed in [supported_datasets.md](docs/supported_datasets.md).

We also share our Kinetics-400 annotation file [k400_val](https://github.com/SwinTransformer/storage/releases/download/v1.0.6/k400_val.txt), [k400_train](https://github.com/SwinTransformer/storage/releases/download/v1.0.6/k400_train.txt) for better comparison.
<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/mmaction2_logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>

  [📘Documentation](https://mmaction2.readthedocs.io/en/latest/) |
  [🛠️Installation](https://mmaction2.readthedocs.io/en/latest/install.html) |
  [👀Model Zoo](https://mmaction2.readthedocs.io/en/latest/modelzoo.html) |
  [🆕Update News](https://mmaction2.readthedocs.io/en/latest/changelog.html) |
  [🚀Ongoing Projects](https://github.com/open-mmlab/mmaction2/projects) |
  [🤔Reporting Issues](https://github.com/open-mmlab/mmaction2/issues/new/choose)
</div>

## Introduction

English | [简体中文](/README_zh-CN.md)

[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/en/latest/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)

MMAction2 is an open-source toolbox for video understanding based on PyTorch.
It is a part of the [OpenMMLab](http://openmmlab.org/) project.

The master branch works with **PyTorch 1.3+**.

<div align="center">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/mmaction2_overview.gif" width="380px"><br>
    <p style="font-size:1.5vw;">Action Recognition Results on Kinetics-400</p>
  </div>
  <div style="float:right;margin-right:0px;">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px"><br>
    <p style="font-size:1.5vw;">Skeleton-base Action Recognition Results on NTU-RGB+D-120</p>
  </div>
</div>
<div align="center">
  <img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="580px"/><br>
    <p style="font-size:1.5vw;">Skeleton-based Spatio-Temporal Action Detection and Action Recognition Results on Kinetics-400</p>
</div>
<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/spatio-temporal-det.gif" width="800px"/><br>
    <p style="font-size:1.5vw;">Spatio-Temporal Action Detection Results on AVA-2.1</p>
</div>

## Major Features

- **Modular design**: We decompose a video understanding framework into different components. One can easily construct a customized video understanding framework by combining different modules.

- **Support four major video understanding tasks**: MMAction2 implements various algorithms for multiple video understanding tasks, including action recognition, action localization, spatio-temporal action detection, and skeleton-based action detection. We support **27** different algorithms and **20** different datasets for the four major tasks.

- **Well tested and documented**: We provide detailed documentation and API reference, as well as unit tests.

## Updates

- (2022-03-04) We support **Multigrid** on Kinetics400, achieve 76.07% Top-1 accuracy and accelerate training speed.
- (2021-11-24) We support **2s-AGCN** on NTU60 XSub, achieve 86.06% Top-1 accuracy on joint stream and 86.89% Top-1 accuracy on bone stream respectively.
- (2021-10-29) We provide a demo for skeleton-based and rgb-based spatio-temporal detection and action recognition (demo/demo_video_structuralize.py).
- (2021-10-26) We train and test **ST-GCN** on NTU60 with 3D keypoint annotations, achieve 84.61% Top-1 accuracy (higher than 81.5% in the [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17135)).
- (2021-10-25) We provide a script(tools/data/skeleton/gen_ntu_rgbd_raw.py) to convert the NTU60 and NTU120 3D raw skeleton data to our format.
- (2021-10-25) We provide a [guide](https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/custom_dataset_training.md) on how to train PoseC3D with custom datasets, [bit-scientist](https://github.com/bit-scientist) authored this PR!
- (2021-10-16) We support **PoseC3D** on UCF101 and HMDB51, achieves 87.0% and 69.3% Top-1 accuracy with 2D skeletons only. Pre-extracted 2D skeletons are also available.

**Release**: v0.22.0 was released in 05/03/2022. Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMAction2.
There are also tutorials:

- [learn about configs](docs/tutorials/1_config.md)
- [finetuning models](docs/tutorials/2_finetune.md)
- [adding new dataset](docs/tutorials/3_new_dataset.md)
- [designing data pipeline](docs/tutorials/4_data_pipeline.md)
- [adding new modules](docs/tutorials/5_new_modules.md)
- [exporting model to onnx](docs/tutorials/6_export_model.md)
- [customizing runtime settings](docs/tutorials/7_customize_runtime.md)

A Colab tutorial is also provided. You may preview the notebook [here](demo/mmaction2_tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb) on Colab.

## Supported Methods

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">Action Recognition</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/c3d/README.md">C3D</a> (CVPR'2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsn/README.md">TSN</a> (ECCV'2016)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/i3d/README.md">I3D</a> (CVPR'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/i3d/README.md">I3D Non-Local</a> (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/r2plus1d/README.md">R(2+1)D</a> (CVPR'2018)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/trn/README.md">TRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md">TSM</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md">TSM Non-Local</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowonly/README.md">SlowOnly</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowfast/README.md">SlowFast</a> (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/csn/README.md">CSN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tin/README.md">TIN</a> (AAAI'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tpn/README.md">TPN</a> (CVPR'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/x3d/README.md">X3D</a> (CVPR'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/omnisource/README.md">OmniSource</a> (ECCV'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition_audio/resnet/README.md">MultiModality: Audio</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tanet/README.md">TANet</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/timesformer/README.md">TimeSformer</a> (ICML'2021)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Action Localization</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/localization/ssn/README.md">SSN</a> (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/localization/bsn/README.md">BSN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/localization/bmn/README.md">BMN</a> (ICCV'2019)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Spatio-Temporal Action Detection</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/acrn/README.md">ACRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/ava/README.md">SlowOnly+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/ava/README.md">SlowFast+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/lfb/README.md">LFB</a> (CVPR'2019)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Skeleton-based Action Recognition</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/stgcn/README.md">ST-GCN</a> (AAAI'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/2s-agcn/README.md">2s-AGCN</a> (CVPR'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/README.md">PoseC3D</a> (ArXiv'2021)</td>
    <td></td>
    <td></td>
  </tr>
</table>

Results and models are available in the *README.md* of each method's config directory.
A summary can be found on the [**model zoo**](https://mmaction2.readthedocs.io/en/latest/recognition_models.html) page.

We will keep up with the latest progress of the community and support more popular algorithms and frameworks.
If you have any feature requests, please feel free to leave a comment in [Issues](https://github.com/open-mmlab/mmaction2/issues/19).

## Supported Datasets

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="4" style="font-weight:bold;">Action Recognition</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/hmdb51/README.md">HMDB51</a> (<a href="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/">Homepage</a>) (ICCV'2011)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ucf101/README.md">UCF101</a> (<a href="https://www.crcv.ucf.edu/research/data-sets/ucf101/">Homepage</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">Homepage</a>) (CVPR'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README.md">Kinetics-[400/600/700]</a> (<a href="https://deepmind.com/research/open-source/kinetics/">Homepage</a>) (CVPR'2017)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/sthv1/README.md">SthV1</a> (<a href="https://20bn.com/datasets/something-something/v1/">Homepage</a>) (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/sthv2/README.md">SthV2</a> (<a href="https://20bn.com/datasets/something-something/">Homepage</a>) (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/diving48/README.md">Diving48</a> (<a href="http://www.svcl.ucsd.edu/projects/resound/dataset.html">Homepage</a>) (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/jester/README.md">Jester</a> (<a href="https://20bn.com/datasets/jester/v1">Homepage</a>) (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/mit/README.md">Moments in Time</a> (<a href="http://moments.csail.mit.edu/">Homepage</a>) (TPAMI'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/mmit/README.md">Multi-Moments in Time</a> (<a href="http://moments.csail.mit.edu/challenge_iccv_2019.html">Homepage</a>) (ArXiv'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/hvu/README.md">HVU</a> (<a href="https://github.com/holistic-video-understanding/HVU-Dataset">Homepage</a>) (ECCV'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/omnisource/README.md">OmniSource</a> (<a href="https://kennymckormick.github.io/omnisource/">Homepage</a>) (ECCV'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/gym/README.md">FineGYM</a> (<a href="https://sdolivia.github.io/FineGym/">Homepage</a>) (CVPR'2020)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">Action Localization</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/thumos14/README.md">THUMOS14</a> (<a href="https://www.crcv.ucf.edu/THUMOS14/download.html">Homepage</a>) (THUMOS Challenge 2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">Homepage</a>) (CVPR'2015)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">Spatio-Temporal Action Detection</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ucf101_24/README.md">UCF101-24*</a> (<a href="http://www.thumos.info/download.html">Homepage</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/jhmdb/README.md">JHMDB*</a> (<a href="http://jhmdb.is.tue.mpg.de/">Homepage</a>) (ICCV'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ava/README.md">AVA</a> (<a href="https://research.google.com/ava/index.html">Homepage</a>) (CVPR'2018)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">Skeleton-based Action Recognition</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-FineGYM</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-NTURGB+D</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-UCF101</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-HMDB51</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
  </tr>
</table>

Datasets marked with * are not fully supported yet, but related dataset preparation steps are provided. A summary can be found on the [**Supported Datasets**](https://mmaction2.readthedocs.io/en/latest/supported_datasets.html) page.

## Benchmark

To demonstrate the efficacy and efficiency of our framework, we compare MMAction2 with some other popular frameworks and official releases in terms of speed. Details can be found in [benchmark](docs/benchmark.md).

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --eval top_k_accuracy

# multi-gpu testing
bash tools/dist_test.sh <CONFIG_FILE> <CHECKPOINT_FILE> <GPU_NUM> --eval top_k_accuracy
```

### Training

To train a video recognition model with pre-trained image models (for Kinetics-400 and Kineticc-600 datasets), run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.backbone.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.backbone.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]
```
For example, to train a `Swin-T` model for Kinetics-400 dataset  with  8 gpus, run:
```
bash tools/dist_train.sh configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py 8 --cfg-options model.backbone.pretrained=<PRETRAIN_MODEL> 
```

To train a video recognizer with pre-trained video models (for Something-Something v2 datasets), run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options load_from=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options load_from=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]
```
For example, to train a `Swin-B` model for SSv2 dataset with 8 gpus, run:
```
bash tools/dist_train.sh configs/recognition/swin/swin_base_patch244_window1677_sthv2.py 8 --cfg-options load_from=<PRETRAIN_MODEL>
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
## FAQ

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Projects built on MMAction2

Currently, there are many research works and projects built on MMAction2 by users from community, such as:

- Video Swin Transformer. [[paper]](https://arxiv.org/abs/2106.13230)[[github]](https://github.com/SwinTransformer/Video-Swin-Transformer)
- Evidential Deep Learning for Open Set Action Recognition, ICCV 2021 **Oral**. [[paper]](https://arxiv.org/abs/2107.10161)[[github]](https://github.com/Cogito2012/DEAR)
- Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective, ICCV 2021 **Oral**. [[paper]](https://arxiv.org/abs/2103.17263)[[github]](https://github.com/xvjiarui/VFS)

etc., check [projects.md](docs/projects.md) to see all related projects.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, use our provided docker or run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, comment out the following code block in the [configuration files](configs/recognition/swin):
```
# do not use mmcv version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Citation
If you find our work useful in your research, please cite:

```
@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}

@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Other Links

> **Image Classification**: See [Swin Transformer for Image Classification](https://github.com/microsoft/Swin-Transformer).

> **Object Detection**: See [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).
MMAction2 is an open-source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features and users who give valuable feedback.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their new models.

> **Semantic Segmentation**: See [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation).

> **Self-Supervised Learning**: See [MoBY with Swin Transformer](https://github.com/SwinTransformer/Transformer-SSL).
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
