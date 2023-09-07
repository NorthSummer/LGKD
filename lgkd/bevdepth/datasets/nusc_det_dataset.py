import os

import mmcv
import numpy as np
import torch
##from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset


from nuscenes import NuScenes
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import view_points, transform_matrix
import os.path as osp

from abc import ABC, abstractmethod
from functools import reduce
from matplotlib.axes import Axes
from pyquaternion import Quaternion
import yaml

__all__ = ['NuscDetDataset']

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


from nuscenes.utils.data_classes import Box, LidarPointCloud

class LidarPointCloud_A(LidarPointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file_multisweep_bf_sample_data(cls,
                                            nusc,
                                            ref_sd_rec,
                                            nsweeps_back,
                                            nsweeps_forward,
                                            return_trans_matrix,
                                            min_distance,
                                            infos,
                                            is_train):

        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame  
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame 
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

        # Aggregate current and previous sweeps.
        current_sd_rec = ref_sd_rec    # "sample_data"
        trans_matrix_list = list()
        skip_frame = 0

        for k in range(nsweeps_back):
            # Load up the pointcloud.

            cur_file_path = osp.join(nusc.dataroot, current_sd_rec['filename'])
            if os.path.exists(cur_file_path):
                current_pc = cls.from_file(cur_file_path)
                #print()
                # Get past pose.
                # 
                current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                # 
                global_from_car = transform_matrix(current_pose_rec['translation'],
                                                Quaternion(current_pose_rec['rotation']), inverse=False)

                
                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                
                current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                
                car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),inverse=False)
                

                # Fuse four transformation matrices into one and perform transform.

                trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                current_pc.transform(trans_matrix)

                trans_matrix_list.append(trans_matrix)
                
                # Remove close points and add timevector.

                current_pc.remove_close(min_distance)

                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # positive difference
        
                if k % (skip_frame + 1) == 0:
                    times = time_lag * np.ones((1, current_pc.nbr_points()))
                else:
                    times = time_lag * np.ones((1, 1))  # dummy value
                all_times = np.hstack((all_times, times))
                

                if k % (skip_frame + 1) == 0:
                    all_pc.points = np.hstack((all_pc.points, current_pc.points))
                else:
                    tmp_points = np.zeros((4, 1), dtype=np.float32)
                    all_pc.points = np.hstack((all_pc.points, tmp_points))  # dummy value

                if current_sd_rec['prev'] == '':  # Abort if there are no previous sweeps.
                    break
                else:
                    current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        #trans_matrix_list = np.stack(trans_matrix_list, axis=0)

        current_sd_rec = ref_sd_rec

        # Abort if there are no future sweeps. Return.
        if current_sd_rec['next'] == '':
            return all_pc, np.squeeze(all_times, axis=0)
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

        for k in range(1, nsweeps_forward + 1):
            # Load up the pointcloud.
            cur_file_path = osp.join(nusc.dataroot, current_sd_rec['filename'])
            if os.path.exists(cur_file_path):
                current_pc = cls.from_file(cur_file_path)

                # Get the pose for this future sweep.

                current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(current_pose_rec['translation'],
                                                Quaternion(current_pose_rec['rotation']), inverse=False)
            
                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)
                
                # Fuse four transformation matrices into one and perform transform.
                trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                current_pc.transform(trans_matrix)

                # Remove close points and add timevector.
                current_pc.remove_close(min_distance)

                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # negative difference

                if k % (skip_frame + 1) == 0:
                    times = time_lag * np.ones((1, current_pc.nbr_points()))
                else:
                    times = time_lag * np.ones((1, 1))  # dummy value
                all_times = np.hstack((all_times, times))

                # Merge with key pc.
                if k % (skip_frame + 1) == 0:
                    all_pc.points = np.hstack((all_pc.points, current_pc.points))
                else:
                    tmp_points = np.zeros((4, 1), dtype=np.float32)
                    all_pc.points = np.hstack((all_pc.points, tmp_points))  # dummy value

                if current_sd_rec['next'] == '':  # Abort if there are no future sweeps.
                    break
                else:
                    current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

        if return_trans_matrix:
            return all_pc, np.squeeze(all_times, 0) #, trans_matrix_list
        else:
            return all_pc, np.squeeze(all_times, 0)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


def bev_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (
            rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    return gt_boxes, rot_mat


def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)


def map_pointcloud_to_image(
    lidar_points,
    img,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(lidar_points.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


class NuscDetDataset(Dataset):

    def __init__(self,
                 ida_aug_conf,
                 bda_aug_conf,
                 classes,
                 data_root,
                 info_paths,
                 is_train,
                 use_cbgs=False,
                 num_sweeps=1,
                 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                               img_std=[58.395, 57.12, 57.375],
                               to_rgb=True),
                 return_depth=False,
                 sweep_idxes=list(),
                 key_idxes=list(),
                 use_fusion=False):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            bda_aug_conf (dict): Config for bda augmentation.
            classes (list): Class names.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            num_sweeps (int): Number of sweeps to be used for each sample.
                default: 1.
            img_conf (dict): Config for image.
            return_depth (bool): Whether to use depth gt.
                default: False.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
            use_fusion (bool): Whether to use lidar data.
                default: False.
        """
        super().__init__()
        if isinstance(info_paths, list):
            self.infos = list()
            for info_path in info_paths:
                self.infos.extend(mmcv.load(info_path))
        else:
            self.infos = mmcv.load(info_paths)
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.data_root = data_root
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.num_sweeps = num_sweeps
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        self.return_depth = return_depth
        assert sum([sweep_idx >= 0 for sweep_idx in sweep_idxes]) \
            == len(sweep_idxes), 'All `sweep_idxes` must greater \
                than or equal to 0.'

        self.sweeps_idx = sweep_idxes
        assert sum([key_idx < 0 for key_idx in key_idxes]) == len(key_idxes),\
            'All `key_idxes` must less than 0.'
        self.key_idxes = [0] + key_idxes
        self.use_fusion = use_fusion
        self.nusc = NuScenes(version='v1.0-trainval',
                        dataroot=data_root,
                        verbose=False)

        self.label_to_reduced = {
            1: 0,
            5: 0,
            7: 0,
            8: 0,
            10: 0,
            11: 0,
            13: 0,
            19: 0,
            20: 0,
            0: 0,
            29: 0,
            31: 0,
            9: 1,
            14: 2,
            15: 3,
            16: 3,
            17: 4,
            18: 5,
            21: 6,
            2: 7,
            3: 7,
            4: 7,
            6: 7,
            12: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            30: 16
        }

        self.label_to_reduced_np = np.zeros(32, dtype=np.int)
        for i in range(32):
            self.label_to_reduced_np[i] = self.label_to_reduced[i]
            
                                    
    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def get_lidar_depth(self, lidar_points, img, lidar_info, cam_info):
        lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
        lidar_ego_pose = lidar_info['LIDAR_TOP']['ego_pose']
        cam_calibrated_sensor = cam_info['calibrated_sensor']
        cam_ego_pose = cam_info['ego_pose']
        pts_img, depth = map_pointcloud_to_image(
            lidar_points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
        return np.concatenate([pts_img[:2, :].T, depth[:, None]],
                              axis=1).astype(np.float32)

    def _fuse_sweep_lidars_v2(self, infos, lidar_info, nsweeps, lidar_calibrated_sensor): 
        #print(lidar_info)
        sample_rec = self.nusc.get('sample_data', lidar_info['data']['LIDAR_TOP'])
        lidar_path = os.path.join("/home/user/BEVDepth/data/nuScenes", sample_rec["filename"])
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        
        lidar_token = lidar_info["data"]["LIDAR_TOP"]
        lidarseg_label_filename = os.path.join("/home/user/BEVDepth/data/nuScenes", self.nusc.get('lidarseg', lidar_token)['filename'])
        y_complete_labels = load_bin_file(lidarseg_label_filename)
        y = self.label_to_reduced_np[y_complete_labels]
                
        y = torch.tensor(y, dtype=torch.long)
        y_mask = (y!=100) #(y!=0) & (y!=11) & (y!=12) & (y!=13) & (y!=14) & (y!=15) 
        #y_mask = y_mask.unsqueeze(1).repeat(1, 4)
        
        lidar_points = LidarPointCloud(points.T)
        
        lidar_points.rotate(
            Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))        
        
        points = lidar_points.points.T
        ###################    
  
        points = torch.FloatTensor(points)
        filt_points = points[y_mask]
        #print(y.shape, points.shape)
        points = np.array(filt_points, dtype=np.float32)
        #print(points.shape)
        
        #lidar_points = LidarPointCloud(points.T)
        
        #agg_lidar_points.rotate(
        #    Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        #agg_lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))        
        
        #sample_rec = self.nusc.get('sample_data', lidar_info['data']['LIDAR_TOP'])
        
        
        #############
        '''
        lidar_points = LidarPointCloud_A(points.T)
        
        agg_lidar_points, _ = lidar_points.from_file_multisweep_bf_sample_data(
                                            self.nusc, 
                                            lidar_info, 
                                            3, 
                                            3, 
                                            False, 
                                            0.5,
                                            infos,
                                            self.is_train) 

        # First step: transform the pointcloud to the ego vehicle
        # frame for the timestamp of the sweep.
    
        #lidar_points = LidarPointCloud(lidar_points.T)
        
        agg_lidar_points.rotate(
            Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        agg_lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))
                                            
        if not type(agg_lidar_points) == np.ndarray:
            
            agg_lidar_points = agg_lidar_points.points.T[:13000,:4]
        '''
        agg_lidar_points = points #lidar_points.points.T
        return agg_lidar_points

    def get_image(self, cam_infos, cams, lidar_infos=None, infos=None):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_depth = list()
        sweep_lidar_depth_fine = list()
        sweep_lidar_points = list()
        sweep_lidar_points_fuse = list()
        for lidar_info in lidar_infos:
            lidar_path = lidar_info['LIDAR_TOP']['filename']
            lidar_points = np.fromfile(os.path.join(self.data_root,
                                                    lidar_path),
                                       dtype=np.float32,
                                       count=-1).reshape(-1, 5)[..., :4]
            sweep_lidar_points.append(lidar_points)
            
            lidar_sample_token = lidar_info['LIDAR_TOP']['sample_token']
            lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
            sample_rec = self.nusc.get('sample', lidar_sample_token)
            #lidar_sample_token = sample_rec
            #sample_rec = self.nusc.get('sample_data',sample_rec['data']['LIDAR_TOP'])            
            fused_lidar_points = self._fuse_sweep_lidars_v2(infos, sample_rec, 3, lidar_calibrated_sensor)
            
            sweep_lidar_points_fuse.append(fused_lidar_points)
            
        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            lidar_depth_fine = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation(
                    )
            for sweep_idx, cam_info in enumerate(cam_infos):

                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                # img = Image.fromarray(img)
                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
                if self.return_depth and (self.use_fusion or sweep_idx == 0):
                    point_depth = self.get_lidar_depth(
                        sweep_lidar_points[sweep_idx], img,
                        lidar_infos[sweep_idx], cam_info[cam])
                        
                    point_depth_fine =  self.get_lidar_depth(
                        sweep_lidar_points_fuse[sweep_idx], img,
                        lidar_infos[sweep_idx], cam_info[cam])
                                                
                    point_depth_augmented = depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    
                    resize_fine = 0.11
                    final_dim_fine = (64, 176)
                    point_depth_fine_augmented = depth_transform(
                        point_depth_fine, resize_fine, final_dim_fine,
                        crop, flip, rotate_ida)
                        
                    #print(point_depth_augmented.shape, point_depth_fine_augmented.shape)  
                        
                    lidar_depth.append(point_depth_augmented)
                    lidar_depth_fine.append(point_depth_fine_augmented)
                    #print(point_depth_fine_augmented.shape)
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))
                sweep_lidar_depth_fine.append(torch.stack(lidar_depth_fine))
                
        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))
            ret_list.append(torch.stack(sweep_lidar_depth_fine).permute(1, 0, 2, 3))
            
        return ret_list

    def get_gt(self, info, cams):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            if (map_name_from_general_to_detection[ann_info['category_name']]
                    not in self.classes
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels)

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.is_train and self.ida_aug_conf['Ncams'] < len(
                self.ida_aug_conf['cams']):
            cams = np.random.choice(self.ida_aug_conf['cams'],
                                    self.ida_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.ida_aug_conf['cams']
        return cams

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        lidar_infos = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]['scene_token'] != self.infos[idx][
                    'scene_token']:
                cur_idx = idx
            info = self.infos[cur_idx]

            
            sample_token = info['lidar_infos']['LIDAR_TOP']['sample_token']            
            sample_info = self.nusc.get('sample', sample_token)
            sample_rec = self.nusc.get('sample_data', sample_info['data']['LIDAR_TOP'])            
            
            cam_infos.append(info['cam_infos'])
            lidar_infos.append(info['lidar_infos'])
            lidar_sweep_timestamps = [
                lidar_sweep['LIDAR_TOP']['timestamp']
                for lidar_sweep in info['lidar_sweeps']
            ]
            for sweep_idx in self.sweeps_idx:
                if len(info['cam_sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                    lidar_infos.append(info['lidar_infos'])
                else:
                    # Handle scenarios when current sweep doesn't have all
                    # cam keys.
                    for i in range(min(len(info['cam_sweeps']) - 1, sweep_idx),
                                   -1, -1):
                        if sum([cam in info['cam_sweeps'][i]
                                for cam in cams]) == len(cams):
                            cam_infos.append(info['cam_sweeps'][i])
                            cam_timestamp = np.mean([
                                val['timestamp']
                                for val in info['cam_sweeps'][i].values()
                            ])
                            # Find the closest lidar frame to the cam frame.
                            lidar_idx = np.abs(lidar_sweep_timestamps -
                                               cam_timestamp).argmin()
                            lidar_infos.append(info['lidar_sweeps'][lidar_idx])
                            break
        lidar_calibrated_sensor = lidar_infos[0]['LIDAR_TOP']['calibrated_sensor']
        fused_lidar_data = self._fuse_sweep_lidars_v2(infos=info, lidar_info=sample_info, nsweeps=10, lidar_calibrated_sensor=lidar_calibrated_sensor) 
        fused_lidar_data = torch.FloatTensor(fused_lidar_data)
        
        if self.return_depth or self.use_fusion:
            image_data_list = self.get_image(cam_infos, cams, lidar_infos)

        else:
            image_data_list = self.get_image(cam_infos, cams, lidar_infos)
        ret_list = list()
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_timestamps,
            img_metas,
        ) = image_data_list[:7]
        img_metas['token'] = self.infos[idx]['sample_token']
        if self.is_train:
            gt_boxes, gt_labels = self.get_gt(self.infos[idx], cams)
        # Temporary solution for test.
        else:
            gt_boxes = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(0, )

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ]
        if self.return_depth:
            ret_list.append(image_data_list[7])
            ret_list.append(image_data_list[8])
            ret_list.append(info)
            ret_list.append(fused_lidar_data)
        
        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)


def collate_fn(data, is_return_depth=False):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    depth_labels_fine_batch = list()
    lidar_batch = list()
    #data_info_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ) = iter_data[:10]
        if is_return_depth:
            gt_depth = iter_data[10]
            gt_depth_fine = iter_data[11]
            data_info = iter_data[12]
            lidar_points = iter_data[13]
            depth_labels_batch.append(gt_depth)
            #print(gt_depth.shape, gt_depth_fine.shape)
            depth_labels_fine_batch.append(gt_depth_fine)
            lidar_batch.append(lidar_points)
            
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
        #data_info_batch.append(data_info)
        
    new_lidar_batch = list()
    #min_num = 0
    for idx, lidar in enumerate(lidar_batch):
        if idx==0:
            min_num = lidar.size(0)
        
        if lidar.size(0) < min_num:
            min_num = lidar.size(0)
    
    for idx, lidar in enumerate(lidar_batch):
        new_lidar_batch.append(lidar[:min_num,:])
        
        
    mats_dict = dict()
    mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
    mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
    mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
    mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
    mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
    ret_list = [
        torch.stack(imgs_batch),
        mats_dict,
        torch.stack(timestamps_batch),
        img_metas_batch,
        gt_boxes_batch,
        gt_labels_batch,
    ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
        ret_list.append(torch.stack(depth_labels_fine_batch))
        ret_list.append(data_info)
        ret_list.append(torch.stack(new_lidar_batch))
        
    return ret_list

