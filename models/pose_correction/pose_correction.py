import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.spatial.transform import Rotation

import models
from .lbs import lbs
# from models.network_utils import get_mlp



def get_transforms_02v(Jtr):
    device = Jtr.device

    from scipy.spatial.transform import Rotation as R
    rot45p = torch.tensor(R.from_euler('z', 45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    rot45n = torch.tensor(R.from_euler('z', -45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)

    # bone_transforms_02v = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(24, 1, 1)
    # do that to cater for smplx ?
    bone_transforms_02v = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(len(Jtr), 1, 1)


    # need to consider the chain in smplx
    
    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    R_02v_l = []
    t_02v_l = []
    chain = [1, 4, 7, 10]
    rot = rot45p
    for i, j_idx in enumerate(chain):
        R_02v_l.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_l[i-1]

        t_02v_l.append(t)

    R_02v_l = torch.stack(R_02v_l, dim=0)
    t_02v_l = torch.stack(t_02v_l, dim=0)
    t_02v_l = t_02v_l - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_l = F.pad(R_02v_l, (0, 0, 0, 1))  # 4 x 4 x 3
    t_02v_l = F.pad(t_02v_l, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_l, t_02v_l.unsqueeze(-1)], dim=-1)

    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    R_02v_r = []
    t_02v_r = []
    chain = [2, 5, 8, 11]
    rot = rot45n
    for i, j_idx in enumerate(chain):
        # bone_transforms_02v[j_idx, :3, :3] = rot
        R_02v_r.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_r[i-1]

        t_02v_r.append(t)

    # bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    R_02v_r = torch.stack(R_02v_r, dim=0)
    t_02v_r = torch.stack(t_02v_r, dim=0)
    t_02v_r = t_02v_r - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_r = F.pad(R_02v_r, (0, 0, 0, 1))  # 4 x 3
    t_02v_r = F.pad(t_02v_r, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_r, t_02v_r.unsqueeze(-1)], dim=-1)

    return bone_transforms_02v

class NoPoseCorrection(nn.Module):
    def __init__(self, config, metadata=None):
        super(NoPoseCorrection, self).__init__()

    def forward(self, camera, iteration):
        return camera, {}

    def regularization(self, out):
        return {}

class PoseCorrection(nn.Module):
    def __init__(self, config, metadata=None):
        super(PoseCorrection, self).__init__()

        self.config = config
        self.metadata = metadata

        self.frame_dict = metadata['frame_dict']

        gender = metadata['gender']

        # need to change the path
        # v_template = np.load('body_models/misc/v_templates.npz')[gender]
        # lbs_weights = np.load('body_models/misc/skinning_weights_all.npz')[gender]
        # posedirs = np.load('body_models/misc/posedirs_all.npz')[gender]
        # posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
        # shapedirs = np.load('body_models/misc/shapedirs_all.npz')[gender]
        # J_regressor = np.load('body_models/misc/J_regressors.npz')[gender]
        # kintree_table = np.load('body_models/misc/kintree_table.npy')

        v_template = metadata['v_templates'][gender]
        # no gender here for skinning weights, already been separated in dataloader
        lbs_weights = metadata['skinning_weights']
        posedirs = metadata['posedirs'][gender]
        posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
        shapedirs = metadata['shapedirs'][gender]
        J_regressor = metadata['J_regressor'][gender]
        kintree_table = metadata['kintree_table']

        self.register_buffer('v_template', torch.tensor(v_template, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('posedirs', torch.tensor(posedirs, dtype=torch.float32))
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32))
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32))
        self.register_buffer('lbs_weights', torch.tensor(lbs_weights, dtype=torch.float32))
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.int32))

    #   need to adjust according to smplx 
    def forward_smpl(self, betas, root_orient, pose_body, pose_hand, trans):
        full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1)
        verts_posed, Jtrs_posed, Jtrs, bone_transforms, _, v_posed, v_shaped, rot_mats = lbs(betas=betas,
                                                                                             pose=full_pose,
                                                                                             v_template=self.v_template.clone(),
                                                                                             clothed_v_template=None,
                                                                                             shapedirs=self.shapedirs.clone(),
                                                                                             posedirs=self.posedirs.clone(),
                                                                                             J_regressor=self.J_regressor.clone(),
                                                                                             parents=self.kintree_table[
                                                                                                 0].long(),
                                                                                             lbs_weights=self.lbs_weights.clone(),
                                                                                             dtype=torch.float32)

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).to(rot_mats.device), rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(1, -1, 9).contiguous()

        bone_transforms_02v = get_transforms_02v(Jtrs.squeeze(0))

        bone_transforms = torch.matmul(bone_transforms.squeeze(0), torch.inverse(bone_transforms_02v))
        bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + trans

        v_shaped = v_shaped.detach()
        center = torch.mean(v_shaped, dim=1)
        minimal_shape_centered = v_shaped - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtrs = Jtrs - center
        Jtrs = (Jtrs - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtrs -= 0.5
        Jtrs *= 2.
        Jtrs = Jtrs.contiguous()

        verts_posed = verts_posed + trans[None]

        return rots, Jtrs, bone_transforms, verts_posed, v_posed, Jtrs_posed

    def forward(self, camera, iteration):
        frame = camera.frame_id
        if frame not in self.frame_dict:
            return camera, {}
        return self.pose_correct(camera, iteration)

    def regularization(self, out):
        raise NotImplementedError

    def pose_correct(self, camera, iteration):
        raise NotImplementedError
    
class DirectPoseOptimization(PoseCorrection):
    def __init__(self, config, metadata=None):
        super(DirectPoseOptimization, self).__init__(config, metadata)
        self.cfg = config

        root_orient = metadata['root_orient']
        pose_body = metadata['pose_body']

        if ('pose_jaw' in metadata.keys()):
            pose_body = np.concatenate([pose_body, metadata['pose_jaw'], metadata['pose_eye']], axis=-1)

        pose_hand = metadata['pose_hand']
        trans = metadata['trans']
        full_betas = metadata['betas']
        frames = metadata['frames']

        self.frames = frames

        # use nn.Embedding
        root_orient = np.array(root_orient)
        pose_body = np.array(pose_body)
        pose_hand = np.array(pose_hand)
        trans = np.array(trans)
        full_betas = np.array(full_betas)

        self.use_expression = full_betas.shape[1] > 10

        if self.use_expression:
            betas = full_betas[0:1, :10]
            expression = full_betas[:, 10:]
            self.expression = nn.Embedding.from_pretrained(torch.from_numpy(expression).float(), freeze=False)
        else:
            betas = full_betas[0:1]

        self.root_orients = nn.Embedding.from_pretrained(torch.from_numpy(root_orient).float(), freeze=False)
        self.pose_bodys = nn.Embedding.from_pretrained(torch.from_numpy(pose_body).float(), freeze=False)
        self.pose_hands = nn.Embedding.from_pretrained(torch.from_numpy(pose_hand).float(), freeze=False)
        self.trans = nn.Embedding.from_pretrained(torch.from_numpy(trans).float(), freeze=False)

        self.register_parameter('betas', nn.Parameter(torch.tensor(betas, dtype=torch.float32)))

    def pose_correct(self, camera, iteration):
        # Correct pose only after a certain number of iterations (5000)
        if iteration < self.cfg.get('delay', 0):
            return camera, {}

        frame = camera.frame_id

        # use nn.Embedding
        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        pose_body = self.pose_bodys(idx)
        pose_hand = self.pose_hands(idx)
        trans = self.trans(idx)

        if self.use_expression:
            expression = self.expression(idx)
            betas = torch.cat([self.betas, expression], dim=-1)
        else:
            betas = self.betas

        # compose rots, Jtrs, bone_transforms, posed_smpl_verts
        rots, Jtrs, bone_transforms, posed_smpl_verts, _, _ = self.forward_smpl(betas, root_orient, pose_body, pose_hand, trans)

        rots_diff = camera.rots - rots
        updated_camera = camera.copy()
        updated_camera.update(
            rots=rots,
            Jtrs=Jtrs,
            bone_transforms=bone_transforms,
        )

        loss_pose = (rots_diff ** 2).mean()
        return updated_camera, {
            'pose': loss_pose,
        }

    def regularization(self, out):
        loss = (out['rots_diff'] ** 2).mean()
        return {'pose_reg': loss}

    def export(self, frame):
        model_dict = {}

        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        pose_body = self.pose_bodys(idx)
        pose_hand = self.pose_hands(idx)
        trans = self.trans(idx)

        if self.use_expression:
            expression = self.expression(idx)
            betas = torch.cat([self.betas, expression], dim=-1)
        else:
            betas = self.betas

        rots, Jtrs, bone_transforms, posed_smpl_verts, v_posed, Jtr_posed = self.forward_smpl(betas, root_orient, pose_body,
                                                                                pose_hand, trans)
        model_dict.update({
            'minimal_shape': v_posed[0],
            'betas': betas,
            'Jtr_posed': Jtr_posed[0],
            'bone_transforms': bone_transforms,
            'trans': trans[0],
            'root_orient': root_orient[0],
            'pose_body': pose_body[0],
            'pose_hand': pose_hand[0],
        })
        for k, v in model_dict.items():
            model_dict.update({k: v.detach().cpu().numpy()})
        return model_dict

def get_pose_correction(cfg, metadata):
    name = cfg.name
    model_dict = {
        "none": NoPoseCorrection,
        "direct": DirectPoseOptimization
    }
    return model_dict[name](cfg, metadata)