# BEV-LGKD: A Unified LiDAR-Guided Knowledge Distillation Framework for Multi-View BEV 3D Object Detection

[![arxiv](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

A standard style for README files


## Background

3D detection is a core technology in autonomous intelligent vehicles, ensuring their perception capability during automated driving. In recent years, many intelligent vehicle manufacturers have adopted surround-view cameras for scene perception, with some even discarding LiDAR and relying solely on surround-view cameras, as demonstrated by Tesla. Our proposed method introduces a novel knowledge distillation approach to multi-view BEV detection, utilizing point cloud information as guidance for the process.

> Recently, the Bird's-Eye-View (BEV) representation has gained increasing attention in multi-view 3D object detection, demonstrating promising applications in autonomous driving. Although multi-view camera-based systems can be deployed at a low cost, the absence of depth information makes current approaches adopt large models for good performance. Therefore, it is essential to improve the efficiency of BEV 3D object detection for practical deployment. Knowledge Distillation (KD) is one of the most practical techniques to train efficient yet accurate models. However, BEV KD is still under-explored to the best of our knowledge. Different from image classification tasks, BEV 3D object detection approaches are more complicated and consist of several components. In this paper, we propose a unified framework named BEV-LGKD to transfer knowledge in a teacher-student manner. However, directly applying the teacher-student paradigm to BEV features fails to achieve satisfying results due to heavy background information in RGB cameras. To solve this problem, we propose to leverage the localization advantage of LiDAR points. Specifically, we transform the LiDAR points into BEV space and generate the view-dependent foreground masks for the teacher-student paradigm. It is noted that our method only uses LiDAR points to guide the KD between RGB models. As the quality of depth estimation is crucial for BEV perception, we further introduce depth distillation to our framework. Our unified framework is simple yet effective and achieves a significant performance boost.
> Remember: the documentation, not the code, defines what a module does.

![framev11.jpg](https://i.imgs.ovh/i/2023/09/06/64f82c9997a26.jpg)













