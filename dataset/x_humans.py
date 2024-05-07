import os
import sys
import glob
import cv2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from scene.cameras import Camera
from utils.camera_utils import freeview_camera

import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh


class X_HumansDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg

        # change in conf
        # ../../data/00036/train/
        self.root_dir = cfg.root_dir

        # Take1
        if split == 'train':
            self.subject = cfg.get('train_subject', 'Take1')
        else:
            # val test predict all group to test
            split = "test"
            self.subject = cfg.get('test_subject', 'Take8')
        self.split = split

        # keep the same?
        self.train_frames = cfg.train_frames
        # need to
        # self.train_cams = cfg.train_views
        self.val_frames = cfg.val_frames
        # self.val_cams = cfg.val_views
        self.white_bg = cfg.white_background
        self.H, self.W = 1024, 1024  # hardcoded original size
        self.h, self.w = cfg.img_hw

        self.model_type = cfg.model_type
        # safely set it as smpl
        if not self.model_type in ['smpl', 'smplx']:
            self.model_type = 'smpl'
        # assert self.model_type in ['smpl', 'smplx']

        # need to cater for SMPLX
        if self.model_type == 'smpl':
            self.faces = np.load('body_models/misc/faces.npz')['faces']
            self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
            self.posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
            self.J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))

            self.v_templates = np.load('body_models/misc/v_templates.npz')
            self.shapedirs = dict(np.load('body_models/misc/shapedirs_all.npz'))
            for k in list(self.shapedirs.keys()):
                self.shapedirs[k] = self.shapedirs[k][:, :, :10]
            self.kintree_table = np.load('body_models/misc/kintree_table.npy')

        elif self.model_type == 'smplx':
            self.faces = np.load('body_models/misc/faces_smplx.npz')['faces']
            self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all_smplx.npz'))
            self.posedirs = dict(np.load('body_models/misc/posedirs_all_smplx.npz'))
            self.J_regressor = dict(np.load('body_models/misc/J_regressors_smplx.npz'))
            # shape of [batch_size, beta_num]
            # self.betas = np.load(os.path.join(self.root_dir, "mean_shape_smplx.npy"))[None, :]

            self.v_templates = np.load('body_models/misc/v_templates_smplx.npz')
            self.shapedirs = dict(np.load('body_models/misc/shapedirs_all_smplx.npz'))
            for k in list(self.shapedirs.keys()):
                self.shapedirs[k] = self.shapedirs[k][:, :, :20]
            self.kintree_table = np.load('body_models/misc/kintree_table_smplx.npy')

        with open(os.path.join(self.root_dir, "gender.txt")) as f:
            self.gender = f.readlines()
        if self.gender not in ['male', 'female', 'neutral']:
            self.gender = 'neutral'

        # if split == 'train':
        #     frames = self.train_frames
        # elif split == 'val':
        #     frames = self.val_frames
        # elif split == 'test':
        #     # frames = self.cfg.test_frames[self.cfg.test_mode]
        #     frames = self.val_frames
        # elif split == 'predict':
        #     frames = self.cfg.predict_frames
        # else:
        #     raise ValueError

        # need camera here
        # ../../data/00036/train/Take1/render/cameras.npz
        self.cameras = []
        self.model_files = []
        self.data = []
        for take in self.subject:
            frames = 0, len(os.listdir(os.path.join(self.root_dir, self.split, take, 'render/image'))), 1
            self.cameras_take = np.load(os.path.join(self.root_dir, self.split, take, 'render/cameras.npz'),
                                allow_pickle=True)
            self.cameras.extend([{'K': self.cameras_take['intrinsic'],
                            'R': self.cameras_take['extrinsic'][k, :3, :3],
                            'T': self.cameras_take['extrinsic'][k, :3, 3]} for k in range(frames[0], frames[1], frames[2])])
            # import ipdb; ipdb.set_trace()

        # zju has one json camera for one scene(has multiple camera view), in the format of {allcameranames:['1',...,], '1':{K:, D:, R:, T:}}
        # for example all images in Coreview_377/1 has same camera setting
        # images with same name are of same pose, Coreview_377/1/00000.jpg and CoreView_377/2/00000.jpg --- to be valided

        # what is D? how to extract it from intrinsic or extrinsic?
        # that dict stored in self.cameras

        # X-humans has train/Take1 as one scene, has ../../data/00036/train/Take1/render/cameras.npz for this view
        # camearas.npz has key 'intrinsic' and 'extrinsic', which are 3*3 matrix and n*4*4 matrix
        # ../../data/00036/train/Take1/render/image/color_000001.png has its own extrinsic camera setting and huamn pose.

        # data

        # if len(cam_names) == 0:
        #     cam_names = self.cameras['all_cam_names']
        # elif self.refine:
        #     cam_names = [f'{int(cam_name) - 1:02d}' for cam_name in cam_names]

            start_frame, end_frame, sampling_rate = frames

            if split == 'predict':
                pass
            # predict_seqs = ['gBR_sBM_cAll_d04_mBR1_ch05_view1',
            #                 'gBR_sBM_cAll_d04_mBR1_ch06_view1',
            #                 'MPI_Limits-03099-op8_poses_view1',
            #                 'canonical_pose_view1',]
            # predict_seq = self.cfg.get('predict_seq', 0)
            # predict_seq = predict_seqs[predict_seq]
            # model_files = sorted(glob.glob(os.path.join(subject_dir, predict_seq, '*.npz')))
            # self.model_files = model_files
            # frames = list(reversed(range(-len(model_files), 0)))
            # if end_frame == 0:
            #     end_frame = len(model_files)
            # frame_slice = slice(start_frame, end_frame, sampling_rate)
            # model_files = model_files[frame_slice]
            # frames = frames[frame_slice]
            else:
                if self.model_type == 'smpl':
                    model_files_take = sorted(
                        glob.glob(os.path.join(self.root_dir, self.split, take, 'SMPL_processed/*.npz')))
                elif self.model_type == 'smplx':
                    model_files_take = sorted(
                        glob.glob(os.path.join(self.root_dir, self.split, take, 'SMPLX_processed/*.npz')))
                # something as [000000.npz, 000001.npz,...,]
                frames = list(range(len(model_files_take)))

                # here config end_frame as files number
                if end_frame == 0:
                    end_frame = len(model_files)
                frame_slice = slice(start_frame, end_frame, sampling_rate)
                self.model_files.extend(model_files_take[frame_slice])
                frames = frames[frame_slice]

            # add freeview rendering
            # init false
            if cfg.freeview:
                # with open(os.path.join(self.root_dir, self.subject, 'freeview_cam_params.json'), 'r') as f:
                #     self.cameras = json.load(f)
                # what is inside data/ZJUMoCap/CoreView_377/models/000000.npz, is it similar to data/00036/train/Take1/SMPL/mesh-f00001_smpl.pkl?
                model_dict = np.load(model_files[0])
                trans = model_dict['transl'].astype(np.float32)
                self.cameras = freeview_camera(self.cameras[0], trans)

            if split == 'predict' or cfg.freeview:
                pass
            # for cam_idx, cam_name in enumerate(cam_names):
            #     cam_dir = os.path.join(subject_dir, cam_name)

            #     for d_idx, f_idx in enumerate(frames):
            #         model_file = model_files[d_idx]
            #         # get dummy gt...
            #         # img_file = glob.glob(os.path.join(cam_dir, '*.jpg'))[0]
            #         img_file = os.path.join(subject_dir, '1', '000000.jpg')
            #         # mask_file = glob.glob(os.path.join(cam_dir, '*.png'))[0]
            #         mask_file = os.path.join(subject_dir, '1', '000000.png')

            #         self.data.append({
            #             'cam_idx': cam_idx,
            #             'cam_name': cam_name,
            #             'data_idx': d_idx,
            #             'frame_idx': f_idx,
            #             'img_file': img_file,
            #             'mask_file': mask_file,
            #             'model_file': model_file,
            #         })
            else:
                # loop over images
                # Only one camera with changing extrinsic parameters
                img_files = sorted(glob.glob(os.path.join(self.root_dir, self.split, take, "render/image/*.png")))[
                    frame_slice]
                mask_files = \
                sorted(glob.glob(os.path.join(self.root_dir, self.split, take, "render/depth/*.tiff")))[frame_slice]
                for d_idx, f_idx in enumerate(frames):
                    img_file = img_files[d_idx]
                    mask_file = mask_files[d_idx]
                    # import ipdb; ipdb.set_trace()
                    model_file = self.model_files[len(self.data)]

                    self.data.append({
                        'data_idx': len(self.data),
                        'frame_idx': f_idx,
                        'img_file': img_file,
                        'mask_file': mask_file,
                        'model_file': model_file
                    })

            self.frames = frames
            # import ipdb; ipdb.set_trace()
            self.get_metadata()

            self.preload = cfg.get('preload', True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]

    # get canonical smpl vertices, need to be smplx now
    # important, used in network
    # check this function later!!
    def get_metadata(self):
        data_paths = self.model_files
        data_path = data_paths[0]

        cano_data = self.get_cano_smpl_verts(data_path)
        if self.split != 'train':
            self.metadata = cano_data
            return

        start, end, step = self.train_frames
        frames = list(range(len(data_paths)))
        if end == 0:
            end = len(frames)
        frame_slice = slice(start, end, step)
        frames = frames[frame_slice]

        frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }

        self.metadata = {
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 3.469298553466797,
            # hardcoded, used to scale the threshold for scaling/image-space gradient
            'frame_dict': frame_dict,
            'v_templates': self.v_templates,
            'shapedirs': self.shapedirs,
            'kintree_table': self.kintree_table,
        }
        self.metadata.update(cano_data)
        if self.cfg.train_smpl:
            self.metadata.update(self.get_smpl_data())

    def get_cano_smpl_verts(self, data_path):
        '''
            Compute star-posed SMPL body vertices.
            To get a consistent canonical space,
            we do not add pose blend shape
        '''
        # compute scale from SMPL body
        gender = self.gender

        model_dict = np.load(data_path, allow_pickle=True)
        minimal_shape = model_dict['minimal_shape']

        # 3D models and points
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights[gender]
        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)

        bone_transforms_02v = get_02v_bone_transforms(Jtr)
        # bone_transforms_02v = np.stack([np.eye(4) for _ in range(len(Jtr))]) 
        # bone transform here is 24, wrong, need to be 55
        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float32)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces)
        return {
            'gender': gender,
            'smpl_verts': vertices.astype(np.float32),
            'minimal_shape': minimal_shape,
            'Jtr': Jtr,
            'skinning_weights': skinning_weights.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v,
            'cano_mesh': cano_mesh,

            'coord_min': coord_min,
            'coord_max': coord_max,
            'aabb': AABB(coord_max, coord_min),
        }

    def get_smpl_data(self):
        # load all smpl fitting of the training sequence
        if self.split != 'train':
            return {}

        from collections import defaultdict
        smpl_data = defaultdict(list)

        for idx, (frame, model_file) in enumerate(zip(self.frames, self.model_files)):
            model_dict = np.load(model_file)

            if idx == 0:
                smpl_data['betas'] = model_dict['betas'][..., : 10].astype(np.float32) 
            # batas change over time now
            smpl_data['expression'].append(model_dict['betas'][..., 10:].astype(np.float32).squeeze(0))


            smpl_data['frames'].append(frame)
            smpl_data['betas'].append(model_dict['betas'].astype(np.float32).reshape(-1))
            smpl_data['root_orient'].append(model_dict['global_orient'].astype(np.float32))
            smpl_data['trans'].append(model_dict['transl'].astype(np.float32))

            if self.model_type == 'smpl':
                smpl_data['pose_body'].append(model_dict['body_pose'][:63].astype(np.float32))
                smpl_data['pose_hand'].append(model_dict['body_pose'][63:].astype(np.float32))
            elif self.model_type == 'smplx':
                smpl_data['pose_hand'].append(
                    np.concatenate([model_dict['left_hand_pose'], model_dict['right_hand_pose']]).astype(np.float32))
                smpl_data['pose_body'].append(
                    np.concatenate([model_dict['body_pose'], model_dict['jaw_pose'], model_dict['leye_pose'],
                                    model_dict['reye_pose']]).astype(np.float32)
                )

        return smpl_data

    def __len__(self):
        return len(self.data)

    def getitem(self, idx, data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]
        data_idx = data_dict['data_idx']
        frame_idx = data_dict['frame_idx']
        img_file = data_dict['img_file']
        mask_file = data_dict['mask_file']
        model_file = data_dict['model_file']

        # import ipdb; ipdb.set_trace()

        K = np.array(self.cameras[data_idx]['K'], dtype=np.float32).copy()
        R = np.array(self.cameras[data_idx]['R'], np.float32)
        T = np.array(self.cameras[data_idx]['T'], np.float32)

        # Todo: Check correctness by projecting
        R = np.transpose(R)

        image = cv2.imread(img_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        # Todo: How is mask used here?

        lanczos = self.cfg.get('lanczos', False)
        interpolation = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_LINEAR

        image = cv2.resize(image, (self.w, self.h), interpolation=interpolation)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        mask = mask != mask.max()
        image[~mask] = 255. if self.white_bg else 0.
        image = image / 255.

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        # update camera parameters
        K[0, :] *= self.w / self.W
        K[1, :] *= self.h / self.H

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # Compute posed SMPL body
        gender = self.metadata['gender']

        model_dict = np.load(model_file, allow_pickle=True)
        minimal_shape = model_dict['minimal_shape']
        n_smpl_points = minimal_shape.shape[0]
        trans = model_dict['transl'].astype(np.float32)
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)
        # Also get GT SMPL poses
        root_orient = model_dict['global_orient'].astype(np.float32)
        pose_body = model_dict['body_pose'][:63].astype(np.float32)
        if self.model_type == 'smpl':
            pose_hand = model_dict['body_pose'][63:].astype(np.float32)
            pose = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
        elif self.model_type == 'smplx':
            pose_hand = np.concatenate([model_dict['left_hand_pose'], model_dict['right_hand_pose']]).astype(np.float32)
            pose_eye = np.concatenate([model_dict['leye_pose'], model_dict['reye_pose']]).astype(np.float32)
            pose_jaw = model_dict['jaw_pose'].astype(np.float32)
            pose = np.concatenate([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], axis=-1)

        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))

        pose_mat_full = pose.as_matrix()  # 24 x 3 x 3
        pose_mat = pose_mat_full[1:, ...].copy()  # 23 x 3 x 3
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
            [-1, 9])  # 24 x 9, root rotation is set to identity
        pose_rot_full = pose_mat_full.reshape([-1, 9])  # 24 x 9, including root rotation

        # Minimally clothed shape
        posedir = self.posedirs[gender]
        Jtr = self.metadata['Jtr']

        # canonical SMPL vertices without pose correction, to normalize joints
        center = np.mean(minimal_shape, axis=0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
        # without global translation
        bone_transforms_02v = self.metadata['bone_transforms_02v']
        bone_transforms = bone_transforms #@ np.linalg.inv(bone_transforms_02v)
        bone_transforms = bone_transforms.astype(np.float32)
        bone_transforms[:, :3, 3] += trans.reshape(1, 3)  # add global offset

        return Camera(
            frame_id=frame_idx,
            K=K, R=R, T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            mask=mask,
            gt_alpha_mask=None,
            image_name=f"f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=self.cfg.data_device,
            # human params
            rots=torch.from_numpy(pose_rot).float().unsqueeze(0),
            Jtrs=torch.from_numpy(Jtr_norm).float().unsqueeze(0),
            bone_transforms=torch.from_numpy(bone_transforms),
        )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)

    # important, used for initialization
    # maybe can directly use those files: data/00036/train/Take1/SMPLX/mesh-f00001_smplx.ply
    def readPointCloud(self, ):
        if self.cfg.get('random_init', False):
            ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')

            aabb = self.metadata['aabb']
            coord_min = aabb.coord_min.unsqueeze(0).numpy()
            coord_max = aabb.coord_max.unsqueeze(0).numpy()
            n_points = 50_000

            xyz_norm = np.random.rand(n_points, 3)
            xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)
        else:
            ply_path = os.path.join(self.root_dir, self.split, self.subject, 'cano_smpl.ply')
            try:
                pcd = fetchPly(ply_path)
            except:
                verts = self.metadata['smpl_verts']
                faces = self.faces
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                n_points = 50_000

                xyz = mesh.sample(n_points)
                rgb = np.ones_like(xyz) * 255
                storePly(ply_path, xyz, rgb)

                pcd = fetchPly(ply_path)

        return pcd


if __name__ == '__main__':
    from omegaconf import OmegaConf

    cfg_dict = {}
    cfg_dict['root_dir'] = '../../data/X_Humans/00036/'
    cfg_dict['split'] = 'train'
    cfg_dict['subject'] = 'Take1'
    cfg_dict['train_frames'] = [0, 100, 1]
    cfg_dict['val_frames'] = [100, 110, 1]
    cfg_dict['white_background'] = False
    cfg_dict['img_hw'] = (1200, 800)
    cfg_dict['model_type'] = 'smplx'
    cfg_dict['freeview'] = False
    cfg_dict['train_smpl'] = False
    cfg_dict['padding'] = 0.1
    cfg_dict['data_device'] = 'cuda'
    cfg_dict['preload'] = False
    cfg_dict['test_mode'] = 'view'
    cfg = OmegaConf.create(cfg_dict)
    dataset = X_HumansDataset(cfg)
    import ipdb;

    ipdb.set_trace()
    res = dataset[0]
