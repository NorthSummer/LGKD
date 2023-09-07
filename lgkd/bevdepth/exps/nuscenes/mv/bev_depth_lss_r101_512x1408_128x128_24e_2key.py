# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch.optim.lr_scheduler import MultiStepLR

from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa
from bevdepth.models.base_bev_depth import BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.backbone_conf['use_da'] = True
        self.data_use_cbgs = True
                
        final_dim = (512, 1408)
        self.backbone_conf['final_dim'] = final_dim
        self.ida_aug_conf['resize_lim'] = (0.386 * 2, 0.55 * 2)
        self.ida_aug_conf['final_dim'] = final_dim
        
        self.backbone_conf_r101 = {
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
                                      depth=101,
                                      frozen_stages=0,
                                      out_indices=[0, 1, 2, 3],
                                      norm_eval=False,
                                      init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
                                  ),
                                    'img_neck_conf':
                                  dict(
                                      type='SECONDFPN',
                                      in_channels=[256, 512, 1024, 2048],
                                      upsample_strides=[0.25, 0.5, 1, 2],
                                      out_channels=[128, 128, 128, 128],
                                  ),
                                  'depth_net_conf':
                                  dict(in_channels=512, mid_channels=512)}
        
        
        self.model = BaseBEVDepth(self.backbone_conf_r101,
                                  self.head_conf,
                                  is_train_depth=True)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [16, 19])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_512x1408_128x128_24e_2key')
