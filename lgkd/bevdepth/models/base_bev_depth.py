import torch
from torch import nn
import torch.nn.functional as F

from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN
from bevdepth.layers.heads.bev_depth_head import BEVDepthHead

from mmcv.ops import Voxelization 
from mmdet3d.models.voxel_encoders import HardSimpleVFE, DynamicVFE, HardVFE, DynamicSimpleVFE
from mmdet3d.models.middle_encoders import SparseEncoder, PointPillarsScatter

__all__ = ['BaseBEVDepth']


class BaseBEVDepth(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super(BaseBEVDepth, self).__init__()
        self.backbone = BaseLSSFPN(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.is_train_depth = is_train_depth

    def forward(
        self,
        x,
        points,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_depth and self.training:
            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            preds = self.head(x)
            return preds, depth_pred, depth_pred, depth_pred, depth_pred, depth_pred, depth_pred
        else:
            x = self.backbone(x, mats_dict, timestamps)
            preds = self.head(x)
            return preds, preds

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
      
        
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.4, 0.4, 0.2]

pts_voxel_layer=dict(
        max_num_points=1,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range)
'''        
pts_voxel_encoder=dict(
        #type='HardSimpleVFE',
        num_features=5,
    )
   
pts_middle_encoder=dict(
        #type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock')

pts_voxel_encoder=dict(
        #type='HardVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01))
''' 

pts_voxel_encoder=dict(
        #type='HardVFE',
        voxel_size=voxel_size,
        point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        )
               
pts_middle_encoder=dict(
        in_channels=1, output_shape=[256, 256])

class BaseBEVDepth_Distill(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf_t, head_conf, backbone_conf_s, is_train_depth=False):
        super(BaseBEVDepth_Distill, self).__init__()
        
        self.backbone = BaseLSSFPN(**backbone_conf_t)
        self.head = BEVDepthHead(**head_conf)

        self.backbone_s = BaseLSSFPN(**backbone_conf_s)
        self.head_s = BEVDepthHead(**head_conf)
        
        self.depth_loss = silog_loss(variance_focus=0.85)
                
        self.is_train_depth = is_train_depth
        
        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.pts_voxel_encoder = DynamicSimpleVFE(**pts_voxel_encoder)
        self.pts_middle_encoder =PointPillarsScatter(**pts_middle_encoder)
        

    def forward(
        self,
        x,
        points,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None. 

        Returns:
            tuple(list[dict]): Output results for tasks. modify head !!!###############
        """
        if self.is_train_depth and self.training:
            
            pts_list = list()
            for pt_idx in range(points.size(0)):
                pt = points[pt_idx, :, :3]
                #print(pt.shape)
                pts_list.append(pt)
                
            voxel_feat = self.extract_pts_feat(pts_list, None, None)
            
            #print(voxel_feat.shape)
            torch.save(voxel_feat, 'bev.pt')
            
            x_t, depth_pred_t, view_mask, fine_d_t = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            
            preds_t = self.head(x_t)

            x_s, depth_pred_s, fine_d_s = self.backbone_s(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
                                          
            preds_s = self.head_s(x_s)            
            
            return preds_t, depth_pred_t, x_t, view_mask, preds_s, depth_pred_s, x_s, fine_d_t, fine_d_s, voxel_feat
            
        else:
            
            x_t, _, _, _ = self.backbone(x, mats_dict, timestamps, is_return_depth=True)
            #print(len(x_t))
            preds_t = self.head(x_t)

            x_s,_,_ = self.backbone_s(x, mats_dict, timestamps, is_return_depth=True)
            preds_s = self.head_s(x_s)
            torch.save(x_s, 'bev_feat.pt')
            return preds_t, preds_s, x_t, x_s

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)

    #@torch.no_grad()
    #@force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        voxel_feats = []
        num_points = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            #print(res_coors[0].shape)
            num_points.append(res_coors[2])
            coors.append(res_coors[1])
            voxel_feats.append(res_coors[0])
            
        points = torch.cat(points, dim=0)
        feats_batch = torch.cat(voxel_feats, dim=0)
        num_points_batch = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            #print(coor.shape)
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        #return points, coors_batch
        return feats_batch, coors_batch, num_points_batch
    
    #@force_fp32()
    def extract_pts_feat(self, points, img_feats, img_metas):
        """Extract point features."""
        #if not self.with_pts_bbox:
        #    return None
        voxels, coors, num_points = self.voxelize(points)
        #print(voxels.shape, coors.shape)
        
        #voxel_features, feature_coors = self.pts_voxel_encoder(
        #    voxels, coors, points)
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels,  coors)
            
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        
        '''
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        '''
        return x