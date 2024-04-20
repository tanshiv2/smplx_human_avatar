import os
import torch
# import trimesh
import glob
import json
import shutil
import argparse

device = torch.device("mps")

import numpy as np
import preprocess_datasets.easymocap.mytools.camera_utils as cam_utils

from scipy.spatial.transform import Rotation

from human_body_prior.body_model.body_model import BodyModel

from preprocess_datasets.easymocap.smplmodel import load_model

parser = argparse.ArgumentParser(
    description='Preprocessing for ZJU-MoCap.'
)
parser.add_argument('--data-dir', type=str, help='Directory that contains raw ZJU-MoCap data.')
parser.add_argument('--out-dir', type=str, help='Directory where preprocessed data is saved.')
parser.add_argument('--seqname', type=str, default='CoreView_313', help='Sequence to process.')

if __name__ == '__main__':
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, seq_name)
    out_dir = os.path.join(args.out_dir, seq_name)

    annots = np.load(os.path.join(data_dir, 'annots.npy'), allow_pickle=True).item()  # cams, ims
    cameras = annots['cams']

    smpl_dir = os.path.join(data_dir, 'new_params')
    verts_dir = os.path.join(data_dir, 'new_vertices')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    body_model = BodyModel(bm_path='body_models/smplx/neutral/model.pkl', num_betas=10, batch_size=1).to(device)

    faces = np.load('body_models/misc/faces.npz')['faces']


    if seq_name in ['CoreView_313', 'CoreView_315']:
        cam_names = list(range(1, 20)) + [22, 23]
    else:
        cam_names = list(range(1, 24))

    cam_names = [str(cam_name) for cam_name in cam_names]

    all_cam_params = {'all_cam_names': cam_names}
    smpl_out_dir = os.path.join(out_dir, 'models') # Preprocessed SMPL models
    if not os.path.exists(smpl_out_dir):
        os.makedirs(smpl_out_dir)

    for cam_idx, cam_name in enumerate(cam_names):
        if seq_name in ['CoreView_313', 'CoreView_315']:
            K = cameras['K'][cam_idx]
            D = cameras['D'][cam_idx]
            R = cameras['R'][cam_idx]
        else:
            K = cameras['K'][cam_idx].tolist() # K is for intrinsic camera parameters
            D = cameras['D'][cam_idx].tolist() # D is for distortion parameters
            R = cameras['R'][cam_idx].tolist() # R is for extrinsic camera parameters

        R_np = np.array(R)
        T = cameras['T'][cam_idx] # T is for translation parameters, in mm
        T_np = np.array(T).reshape(3, 1) / 1000.0
        T = T_np.tolist()

        cam_out_dir = os.path.join(out_dir, cam_name)
        if not os.path.exists(cam_out_dir):
            os.makedirs(cam_out_dir)

        if seq_name in ['CoreView_313', 'CoreView_315']:
            img_in_dir = os.path.join(data_dir, 'Camera ({})'.format(cam_name))
            mask_in_dir = os.path.join(data_dir, 'mask_cihp/Camera ({})'.format(cam_name))
        else:
            img_in_dir = os.path.join(data_dir, 'Camera_B{}'.format(cam_name))
            mask_in_dir = os.path.join(data_dir, 'mask_cihp/Camera_B{}'.format(cam_name))

        img_files = sorted(glob.glob(os.path.join(img_in_dir, '*.jpg')))

        cam_params = {'K': K, 'D': D, 'R': R, 'T': T}
        all_cam_params.update({cam_name: cam_params})

        for img_file in img_files:
            print ('Processing: {}'.format(img_file))
            if seq_name in ['CoreView_313', 'CoreView_315']:
                idx = int(os.path.basename(img_file).split('_')[4])
                frame_index = idx - 1
            else:
                idx = int(os.path.basename(img_file)[:-4])
                frame_index = idx

            # Each frame has a corresponding mask, SMPL parameters, and vertices
            # SMPL parameters are stored in new_params directory
            # Vertices are stored in new_vertices directory
            
            mask_file = os.path.join(mask_in_dir, os.path.basename(img_file)[:-4] + '.png')
            smpl_file = os.path.join(smpl_dir, '{}.npy'.format(idx))
            verts_file = os.path.join(verts_dir, '{}.npy'.format(idx))

            if not os.path.exists(smpl_file):
                print ('Cannot find SMPL file for {}: {}, skipping'.format(img_file, smpl_file))
                continue

            # We only process SMPL parameters in world coordinate
            if cam_idx == 0:
                params = np.load(smpl_file, allow_pickle=True).item()
                # params is a dictionary with keys: 'Rh', 'Th', 'shapes', 'poses'
                # 'Rh' is the root orientation in axis-angle representation
                # 'Th' is the translation of the root joint
                # 'shapes' is the shape parameters representing shape blend shapes, which are the coefficients of the shape PCA basis
                # 'poses' is the pose parameters representing pose blend shapes in axis-angle form
                # The first 3 values in 'poses' are the global rotation of the root joint, 3-66 are the body pose, and 66- are the hand pose (totally 23 joints)
                # For smplx model, the first 3 values in 'poses' are the global rotation of the root joint, 3-66 are the body pose, 66-156 are the hand pose, and 156-162 are the eye pose, and 162-165 are the jaw pose

                root_orient = Rotation.from_rotvec(np.array(params['Rh']).reshape([-1])).as_matrix()
                trans = np.array(params['Th']).reshape([3, 1])

                betas = np.array(params['shapes'], dtype=np.float32)
                import ipdb; ipdb.set_trace()
                poses = np.array(params['poses'], dtype=np.float32)
                pose_body = poses[:, 3:66].copy()
                pose_hand = poses[:, 66:].copy()
                if body_model.model_type == 'smplx':
                    if poses.shape[1] == 165:
                        pose_hand = poses[:, 66:156].copy()
                    else:
                        pose_hand = np.zeros([1, 90], dtype=np.float32)
                    

                pose_jaw = np.zeros([1, 3], dtype=np.float32)
                pose_eye = np.zeros([1, 6], dtype=np.float32)
                if body_model.model_type == 'smplx':
                    if poses.shape[1] == 165:
                        pose_jaw = poses[:, 162:165].copy()
                        pose_eye = poses[:, 156:162].copy()
                        

                if body_model.model_type == 'smplx':
                    poses = np.concatenate([poses[:,:3], pose_body, pose_hand, pose_eye, pose_jaw], axis=1).astype(np.float32)
                poses_torch = torch.from_numpy(poses).to(device)
                pose_body_torch = torch.from_numpy(pose_body).to(device)
                pose_hand_torch = torch.from_numpy(pose_hand).to(device)
                pose_jaw_torch = torch.from_numpy(pose_jaw).to(device)
                pose_eye_torch = torch.from_numpy(pose_eye).to(device)
                betas_torch = torch.from_numpy(betas).to(device)

                new_root_orient = Rotation.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
                new_trans = trans.reshape([1, 3]).astype(np.float32)

                new_root_orient_torch = torch.from_numpy(new_root_orient).to(device)
                new_trans_torch = torch.from_numpy(new_trans).to(device)

                # Get shape vertices
                body = body_model(betas=betas_torch)
                minimal_shape = body.v.detach().cpu().numpy()[0]
                # There are 6890 vertices in the SMPL model
                # minimal_shape is the shape of the SMPL model with the given shape parameters, without any pose applied

                # Get bone transforms
                if body_model.model_type == 'smpl':
                    body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)
                elif body_model.model_type == 'smplx':
                    body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch, pose_jaw=pose_jaw_torch, pose_eye=pose_eye_torch)
                # Global rotation is included when calculating LBS

                if body_model.model_type == 'smpl':
                    body_model_em = load_model(gender='neutral', model_type='smpl').to(device)
                elif body_model.model_type == 'smplx':
                    body_model_em = load_model(gender='neutral', model_type='smplx').to(device)
                
                verts = body_model_em(poses=poses_torch, shapes=betas_torch, Rh=new_root_orient_torch, Th=new_trans_torch, return_verts=True)[0].detach().cpu().numpy()
                # Global rotation is not included when calculating LBS. It is applied to vertices after LBS

                vertices = body.v.detach().cpu().numpy()[0]

                new_trans = new_trans + (verts - vertices).mean(0, keepdims=True)


                new_trans_torch = torch.from_numpy(new_trans).to(device)

                body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)
                # Calculate LBS with the new translation

                # # Visualize SMPL mesh
                # vertices = body.v.detach().cpu().numpy()[0]
                # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                # out_filename = os.path.join(smpl_out_dir, '{:06d}.ply'.format(idx))
                # mesh.export(out_filename)

                import ipdb; ipdb.set_trace()
                bone_transforms = body.bone_transforms.detach().cpu().numpy()
                Jtr_posed = body.Jtr.detach().cpu().numpy()

                out_filename = os.path.join(smpl_out_dir, '{:06d}.npz'.format(idx))

                np.savez(out_filename,
                         minimal_shape=minimal_shape,
                         betas=betas,
                         Jtr_posed=Jtr_posed[0],
                         bone_transforms=bone_transforms[0],
                         trans=new_trans[0],
                         root_orient=new_root_orient[0],
                         pose_body=pose_body[0],
                         pose_hand=pose_hand[0])

            shutil.copy(os.path.join(img_file), os.path.join(cam_out_dir, '{:06d}.jpg'.format(idx)))
            shutil.copy(os.path.join(mask_file), os.path.join(cam_out_dir, '{:06d}.png'.format(idx)))

    with open(os.path.join(out_dir, 'cam_params.json'), 'w') as f:
        json.dump(all_cam_params, f)
