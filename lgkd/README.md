# BEV-LGKD: A Unified LiDAR-Guided Knowledge Distillation Framework for Multi-View BEV 3D Object Detection

[![arxiv](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

A standard style for README files




## Contents

- [Background](#background)
- [Install](#install)
- [Checkpoints](#checkpoints)
- [Usage](#usage)
	- [Inference](#inference)


## Background

3D detection is a core technology in autonomous intelligent vehicles, ensuring their perception capability during automated driving. In recent years, many intelligent vehicle manufacturers have adopted surround-view cameras for scene perception, with some even discarding LiDAR and relying solely on surround-view cameras, as demonstrated by Tesla. Our proposed method introduces a novel knowledge distillation approach to multi-view BEV detection, utilizing point cloud information as guidance for the process.

> Recently, the Bird's-Eye-View (BEV) representation has gained increasing attention in multi-view 3D object detection, demonstrating promising applications in autonomous driving. Although multi-view camera-based systems can be deployed at a low cost, the absence of depth information makes current approaches adopt large models for good performance. Therefore, it is essential to improve the efficiency of BEV 3D object detection for practical deployment. Knowledge Distillation (KD) is one of the most practical techniques to train efficient yet accurate models. However, BEV KD is still under-explored to the best of our knowledge. Different from image classification tasks, BEV 3D object detection approaches are more complicated and consist of several components. In this paper, we propose a unified framework named BEV-LGKD to transfer knowledge in a teacher-student manner. However, directly applying the teacher-student paradigm to BEV features fails to achieve satisfying results due to heavy background information in RGB cameras. To solve this problem, we propose to leverage the localization advantage of LiDAR points. Specifically, we transform the LiDAR points into BEV space and generate the view-dependent foreground masks for the teacher-student paradigm. It is noted that our method only uses LiDAR points to guide the KD between RGB models. As the quality of depth estimation is crucial for BEV perception, we further introduce depth distillation to our framework. Our unified framework is simple yet effective and achieves a significant performance boost.
> Remember: the documentation, not the code, defines what a module does.

![framev11.jpg](https://i.imgs.ovh/i/2023/09/06/64f82c9997a26.jpg)





## Install

Please make sure that your CUDA version supports the following versions of dependencies:

**Step 1:** Install PyTorch
```sh
$ pip install torch==2.0.0 
```

**Step 2:** Install OpenMMLab-dependencies
```sh
$ pip install mmcv==2.0.0rc1
$ pip install mmdet==3.0.0rc3
$ pip install mmcls==1.0.0rc3
$ pip install mmsegmentatio==1.0.0rc0
$ pip install mmdet3d==1.1.0rc2
$ pip install mmengine==0.8.4
```

**Step 3:** Install other dependencies
```sh
$ pip install -r requirements.txt
```

**Step 4:** Build the codebase
```sh
$ cd ../LGKD
$ python setup.py develop
$ mkdir ./data
$ ln -s <your source path of nuscenes dataset> <./data/nuScenes>
$ python scripts/gen_info.py
```



## Checkpoints


| Student Type | Input Size  |mAP  | NDS | mATE | Checkpoint_google| Checkpoint_baidu (soon)|
| :------------:|:---------------:|:---------------:|:---------------:|:-----:|:-----:|:-----:|
| Res-18     | 256x704 |0.305 | 0.435 | 0.711 |[Google Drive](https://drive.google.com/file/d/10YJ7SrmmHtbQ86ST3aBWqtNuLx5YFcjt/view?usp=drive_link) |[Baidu C-Disk]() |
| Res-50     | 256x704 |0.340 | 0.465 | 0.643 |[Google Drive](https://drive.google.com/file/d/1EQt_kj54PhxRQGj0tzWToarckFelP4Je/view?usp=sharing) |[Baidu C-Disk]() |
| ConvNeXt      |256x704 |0.372 |   0.485 | 0.616  |[Google Drive](https://drive.google.com/file/d/1m36jZ5DUAUxAxbCFXWvUfH29PDp1wDDR/view?usp=drive_link) |[Baidu C-Disk]() |
| ConvNeXt-CBGS      |256x704 |0.386 |   0.502| 0.586 |[Google Drive](https://drive.google.com/file/d/1vTgzdDUk614HmhHvyiI-LHNe6U6VF4M3/view?usp=drive_link) |[Baidu C-Disk]() |

## Usage

### Train
The training codes will be updated in few days.
### Inference
To evaluate the performance of distilled model on nuScenes Val set:

```sh
$ python <exp_name> -e --gpus 1 -b 4 --ckpt_path <Your checkpoint path>

```
You may modify <exp_name> as bevdepth/exps/nuscenes/mv/lgkd_convnext_256x704_256x256_20e_cbgs_2key.py

## Reference
For more details, please refer to our paper:
```
@article{li2023bev,
  title={BEV-LGKD: A Unified LiDAR-Guided Knowledge Distillation Framework for Multi-View BEV 3D Object Detection},
  author={Li, Jianing and Lu, Ming and Liu, Jiaming and Guo, Yandong and Du, Yuan and Du, Li and Zhang, Shanghang},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2023},
  publisher={IEEE}
}
```


