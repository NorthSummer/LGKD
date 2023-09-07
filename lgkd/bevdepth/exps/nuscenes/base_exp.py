# Copyright (c) Megvii Inc. All rights reserved.
import os
from functools import partial

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from bevdepth.evaluators.det_evaluators import DetNuscEvaluator
from bevdepth.models.base_bev_depth import BaseBEVDepth, BaseBEVDepth_Distill
from bevdepth.utils.torch_dist import all_gather_object, get_rank, synchronize
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

import mmcls
import mmpretrain
import mmdet


H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.4],
    'y_bound': [-51.2, 51.2, 0.4],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
      type='mmpretrain.ConvNeXt', # Add mmpretrain prefix to enable cross-library mechanism
      arch='base',
      out_indices=[0, 1, 2, 3],
      drop_path_rate=0.5,
      #layer_scale_init_value=1.0,
      gap_before_final_norm=False,
      init_cfg=dict(
          type='Pretrained',
          checkpoint=
          'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128-noema_in1k_20220222-dba4f95f.pth',
          prefix='backbone.')),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[128, 256, 512, 1024],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512),
    'teacher': True
}


backbone_conf_s = {
    'x_bound': [-51.2, 51.2, 0.4],
    'y_bound': [-51.2, 51.2, 0.4],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
      type='mmpretrain.ConvNeXt', # Add mmpretrain prefix to enable cross-library mechanism
      arch='base',
      out_indices=[0, 1, 2, 3],
      drop_path_rate=0.5,
      #layer_scale_init_value=1.0,
      gap_before_final_norm=False,
      init_cfg=dict(
          type='Pretrained',
          checkpoint=
          'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128-noema_in1k_20220222-dba4f95f.pth',
          prefix='backbone.')),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[128, 256, 512, 1024],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512),
    'teacher': False
}

backbone_conf_s_r18 = {
    'x_bound': [-51.2, 51.2, 0.4],
    'y_bound': [-51.2, 51.2, 0.4],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=18,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256, 512],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512),
    'teacher':False
}


backbone_conf_base = {
    'x_bound': [-51.2, 51.2, 0.4],
    'y_bound': [-51.2, 51.2, 0.4],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=18,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256, 512],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512),
    'teacher':False
}

ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim':
    final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}

bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4, #4
    voxel_size=[0.1, 0.1, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[1024, 1024, 1],
    voxel_size=[0.1, 0.1, 8],
    out_size_factor=4, #4
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.1, 0.1, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(1/depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
        


class BEVDepthLightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                 gpus: int = 1,
                 data_root='data/nuScenes',
                 eval_interval=1,
                 batch_size_per_device=8,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 bda_aug_conf=bda_aug_conf,
                 default_root_dir='./outputs/',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.basic_lr_per_img = 2e-4 / 64
        self.class_names = class_names
        self.backbone_conf_s = backbone_conf_s
        self.backbone_conf_t = backbone_conf
        self.backbone_conf = backbone_conf_base
        self.head_conf = head_conf
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        #mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetNuscEvaluator(class_names=self.class_names,
                          output_dir=self.default_root_dir)
                          
        self.model = BaseBEVDepth_Distill(self.backbone_conf_t,
                                 self.head_conf,
                                  self.backbone_conf_s,
                                  is_train_depth=True)
                                 
        '''                                 
        self.model = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)
        
        '''
        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.data_return_depth = True
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.dbound = self.backbone_conf['d_bound']
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.use_fusion = False
        self.train_info_paths = os.path.join(self.data_root,
                                             'nuscenes_infos_train.pkl')
        self.val_info_paths = os.path.join(self.data_root,
                                           'nuscenes_infos_val.pkl')
        self.predict_info_paths = os.path.join(self.data_root,
                                               'nuscenes_infos_test.pkl')
        self.sparsity = 0
                
    def forward(self, sweep_imgs, lidar_points, mats):
        return self.model(sweep_imgs, lidar_points, mats)

    def training_step(self, batch):
        #(sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = batch
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels, depth_fine_labels, _, lidar_points) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]

        preds_t, depth_preds_t, bev_feat_t, view_masks, preds_s, depth_preds_s, bev_feat_s, depth_preds_f_t, depth_preds_f_s, lidar_mask = self(sweep_imgs, lidar_points, mats)
        #preds_s, depth_preds_s = self(sweep_imgs, mats)
         
        preds = preds_s
        depth_preds = depth_preds_s


        non_zero_count = torch.nonzero(view_masks[0]).size(0)
        total_elements = view_masks[0].numel()
        non_zero_ratio = non_zero_count / total_elements        
        
        self.sparsity = self.sparsity + non_zero_ratio
        print(self.sparsity) 
        
        depth_preds_t = depth_preds_t.detach()
        bev_feat_t = bev_feat_t.detach()
        depth_preds_f_t = depth_preds_f_t.detach()
        

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)

        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
        self.log('detection_loss', detection_loss)
        self.log('depth_loss', depth_loss)
        
        depth_fine_loss = self.get_depth_fine_loss(depth_preds_f_s, depth_fine_labels)
        
        T = 1.25
        l_bev = self.bev_distillation_loss(bev_feat_s, bev_feat_t, view_masks, lidar_mask)
        
        l_cd = self.c_depth_distillation_loss(depth_preds_s, depth_preds_t, depth_labels, T)
        
        l_soft = self.soft_distillation_loss(preds_s, preds_t, T)
        
        #print(l_bev, l_cd, l_soft)
        
        return detection_loss + depth_loss + l_bev + l_cd + l_soft #+ depth_fine_loss

    def get_depth_fine_loss(self, depth_preds_fine, depth_labels_fine):
        
        depth_preds_fine = depth_preds_fine.squeeze(1)
        depth_labels_fine = depth_labels_fine.squeeze(1).reshape(6, 64, 176)
        
        fine_depth_loss = silog_loss(variance_focus=0.85)
        fg_mask = (depth_labels_fine > 0.001) & (depth_labels_fine < 100.0) & (depth_preds_fine > 0.001) & (depth_preds_fine < 100.0) #torch.max(depth_labels, dim=1).values > 0.0
             
        assert depth_labels_fine.size() == depth_preds_fine.size()

        depth_fine_loss = fine_depth_loss(depth_preds_fine, depth_labels_fine , fg_mask)
                   
        return 3.0 * depth_fine_loss
        
        
    def ce_distill_loss(self, pred, tar, T):
        loss = nn.KLDivLoss()(F.log_softmax(pred / T, dim=1), F.softmax(tar / T, dim=1)) * (T * T)
        return loss

    

    def bev_distillation_loss(self, bev_feat_s, bev_feat_t, view_list, lidar_mask):    
        
        assert bev_feat_s.size() == bev_feat_t.size()
        
        B, C, H, W = bev_feat_s.size()
        lidar_mask = lidar_mask.repeat(1, C, 1, 1)        
        for idx, view in enumerate(view_list):
             
            view = torch.sum(view, dim=1).unsqueeze(1).repeat(B, C, 1, 1) 

            mask = (bev_feat_t != 0) & (view != 0) & (lidar_mask != 0)  #bev_feat_t[bev_feat_t != 0]
            
            if idx == 0:          
                loss_sum = F.mse_loss(bev_feat_s[mask], bev_feat_t[mask])                
            else:                               
                loss_sum =  loss_sum + F.mse_loss(bev_feat_s[mask], bev_feat_t[mask])
            
        return loss_sum
        
    def c_depth_distillation_loss(self, c_depth_s, c_depth_t, depth_labels, T):
    
        c_depth_s = c_depth_s.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)

        c_depth_t = c_depth_t.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)            
            
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        
        with autocast(enabled=False):
            #print(c_depth_t)
            loss = (T * T * F.binary_cross_entropy(c_depth_s[fg_mask]/T, c_depth_t[fg_mask]/T, reduction='none').sum() / max(1.0, fg_mask.sum()))
        
        #self.ce_distill_loss(c_depth_s[fg_mask], c_depth_t[fg_mask], T) #T * T * F.cross_entropy(c_depth_s/T, c_depth_t /T)
        
        return loss

    def get_heatmap_mask(self, preds_t):
    
        heat_map_list = list()       
        for idx, pred_t in enumerate(preds_t):           
                heat_map_list.append(pred_t[0]['heatmap'].detach())
        
        return heat_map_list
            
    def soft_distillation_loss(self, preds_s, preds_t, T):
        
        for idx, (pred_s, pred_t) in enumerate(zip(preds_s, preds_t)):
        
            if idx == 0:
                loss_sem =  self.ce_distill_loss(pred_s[0]['heatmap'], pred_t[0]['heatmap'].detach(), T) #T * T * F.cross_entropy(pred_s[0]['heatmap']/T, pred_t[0]['heatmap']/T)
                loss_reg =  F.smooth_l1_loss(pred_s[0]['reg'], pred_t[0]['reg'].detach())
                loss_rot =  F.smooth_l1_loss(pred_s[0]['rot'], pred_t[0]['rot'].detach())
            else:    
                loss_sem = loss_sem + self.ce_distill_loss(pred_s[0]['heatmap'], pred_t[0]['heatmap'].detach(), T) #T * T * F.cross_entropy(pred_s[0]['heatmap']/T, pred_t[0]['heatmap']/T)
                loss_reg = loss_reg + F.smooth_l1_loss(pred_s[0]['reg'], pred_t[0]['reg'].detach())
                loss_rot = loss_rot + F.smooth_l1_loss(pred_s[0]['rot'], pred_t[0]['rot'].detach())
              
    
        return 0.25 * loss_sem + loss_reg + loss_rot
        
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds_t, preds_s, bev_t, bev_s = self.model(sweep_imgs, mats, mats) 
        
        non_zero_count = torch.nonzero(bev_s).size(0)
        
        total_elements = bev_s.numel()
        non_zero_ratio = non_zero_count / total_elements        
        
        self.sparsity = self.sparsity + non_zero_ratio
        #print(self.sparsity)
        
        
        preds = preds_s
        
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])
        synchronize()
        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def test_epoch_end(self, test_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()
        # TODO: Change another way.
        dataset_length = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:dataset_length]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]

    def train_dataloader(self):
        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                       bda_aug_conf=self.bda_aug_conf,
                                       classes=self.class_names,
                                       data_root=self.data_root,
                                       info_paths=self.train_info_paths,
                                       is_train=True,
                                       use_cbgs=self.data_use_cbgs,
                                       img_conf=self.img_conf,
                                       num_sweeps=self.num_sweeps,
                                       sweep_idxes=self.sweep_idxes,
                                       key_idxes=self.key_idxes,
                                       return_depth=self.data_return_depth,
                                       use_fusion=self.use_fusion)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                     bda_aug_conf=self.bda_aug_conf,
                                     classes=self.class_names,
                                     data_root=self.data_root,
                                     info_paths=self.val_info_paths,
                                     is_train=False,
                                     img_conf=self.img_conf,
                                     num_sweeps=self.num_sweeps,
                                     sweep_idxes=self.sweep_idxes,
                                     key_idxes=self.key_idxes,
                                     return_depth=self.use_fusion,
                                     use_fusion=self.use_fusion)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion),
            num_workers=4,
            sampler=None,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        predict_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                         bda_aug_conf=self.bda_aug_conf,
                                         classes=self.class_names,
                                         data_root=self.data_root,
                                         info_paths=self.predict_info_paths,
                                         is_train=False,
                                         img_conf=self.img_conf,
                                         num_sweeps=self.num_sweeps,
                                         sweep_idxes=self.sweep_idxes,
                                         key_idxes=self.key_idxes,
                                         return_depth=self.use_fusion,
                                         use_fusion=self.use_fusion)
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion),
            num_workers=4,
            sampler=None,
        )
        return predict_loader

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'predict')

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser
