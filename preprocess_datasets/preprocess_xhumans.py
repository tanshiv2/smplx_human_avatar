import torch
import os
import numpy as np
import glob
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from human_body_prior.body_model.body_model import BodyModel
from preprocess_datasets.easymocap.smplmodel import load_model

if __name__ == '__main__':
    root_dir = '../../data/X_Humans/00036/'
    split = 'train'
    model_type = 'smplx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gender = 'female'
    if model_type == 'smplx':
        bm_path = 'body_models/smplx/' + gender + '/model.pkl'
        # Extract minimal shape for smplx
        body_model = BodyModel(bm_path=bm_path, num_betas=20, batch_size=1).to(device)

        all_takes = glob.glob(os.path.join(root_dir, split, '*'))
        for take_dir in all_takes:
            print("Processing {}".format(take_dir))
            if not os.path.exists(os.path.join(take_dir, 'SMPLX_processed')):
                os.makedirs(os.path.join(take_dir, 'SMPLX_processed'))
            smpl_files = glob.glob(os.path.join(take_dir, 'SMPLX', '*.pkl'))
            pointcloud_files = glob.glob(os.path.join(take_dir, 'SMPLX', '*.ply'))
            assert len(smpl_files) == len(pointcloud_files)
            for smpl_file, pointcloud_file in tqdm(zip(smpl_files, pointcloud_files), total=len(smpl_files)):
                params = np.load(smpl_file, allow_pickle=True)

                root_orient = Rotation.from_rotvec(np.array(params['global_orient']).reshape([-1])).as_matrix()
                trans = np.array(params['transl']).reshape(3, 1)
                betas = np.array(params['betas'], dtype=np.float32).reshape(1, -1)
                expression = np.array(params['expression'], dtype=np.float32).reshape(1, -1)
                full_betas = np.concatenate([betas, expression], axis=-1)

                pose_body = np.array(params['body_pose'], dtype=np.float32).reshape(1, -1)
                pose_lefthand = np.array(params['left_hand_pose'], dtype=np.float32).reshape(1, -1)
                pose_righthand = np.array(params['right_hand_pose'], dtype=np.float32).reshape(1, -1)
                pose_lefteye = np.array(params['leye_pose'], dtype=np.float32).reshape(1, -1)
                pose_righteye = np.array(params['reye_pose'], dtype=np.float32).reshape(1, -1)
                pose_jaw = np.array(params['jaw_pose'], dtype=np.float32).reshape(1, -1)

                poses = np.concatenate(
                    [np.zeros((1, 3)), pose_body, pose_jaw, pose_lefteye, pose_righteye,
                     pose_lefthand, pose_righthand], axis=-1).astype(np.float32)

                poses_torch = torch.from_numpy(poses).to(device)
                pose_body_torch = torch.from_numpy(pose_body).to(device)
                pose_left_hand_torch = torch.from_numpy(pose_lefthand).to(device)
                pose_right_hand_torch = torch.from_numpy(pose_righthand).to(device)
                pose_left_eye_torch = torch.from_numpy(pose_lefteye).to(device)
                pose_right_eye_torch = torch.from_numpy(pose_righteye).to(device)
                pose_jaw_torch = torch.from_numpy(pose_jaw).to(device)
                betas_torch = torch.from_numpy(betas).to(device)
                expression_torch = torch.from_numpy(expression).to(device)
                full_betas_torch = torch.from_numpy(full_betas).to(device)

                new_root_orient = Rotation.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
                new_trans = trans.reshape([1, 3]).astype(np.float32)

                new_root_orient_torch = torch.from_numpy(new_root_orient).to(device)
                new_trans_torch = torch.from_numpy(new_trans).to(device)

                body = body_model(betas=full_betas_torch)
                import ipdb; ipdb.set_trace()
                minimal_shape = body.v.detach().cpu().numpy()[0]

                pose_hand_torch = torch.cat([pose_left_hand_torch, pose_right_hand_torch], axis=-1)
                pose_eye_torch = torch.cat([pose_left_eye_torch, pose_right_eye_torch], axis=-1)

                # Get bone transforms
                body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch,
                                  pose_hand=pose_hand_torch, betas=full_betas_torch, trans=new_trans_torch,
                                  pose_jaw=pose_jaw_torch, pose_eye=pose_eye_torch)

                body_model_em = load_model(gender=gender, model_type='smplx').to(device)
                verts = body_model_em(poses=poses_torch, shapes=betas_torch, expression=expression_torch,
                                      Rh=new_root_orient_torch, Th=new_trans_torch,
                                      return_verts=True)[0].detach().cpu().numpy()

                vertices = body.v.detach().cpu().numpy()[0]

                new_trans = new_trans + (verts - vertices).mean(0, keepdims=True)

                new_trans_torch = torch.from_numpy(new_trans).to(device)
                body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch,
                                  pose_hand=pose_hand_torch, betas=full_betas_torch, trans=new_trans_torch,
                                  pose_jaw=pose_jaw_torch, pose_eye=pose_eye_torch)
                bone_transforms = body.bone_transforms.detach().cpu().numpy()
                Jtr_posed = body.Jtr.detach().cpu().numpy()

                file_name = smpl_file.split('/')[-1]

                out_filename = os.path.join(take_dir, 'SMPLX_processed', file_name)
                np.savez(out_filename,
                         minimal_shape=minimal_shape,
                         betas=betas,
                         Jtr_posed=Jtr_posed[0],
                         bone_transforms=bone_transforms[0],
                         transl=new_trans[0],
                         global_orient=new_root_orient[0],
                         body_pose=pose_body[0],
                         jaw_pose=pose_jaw[0],
                         leye_pose=pose_lefteye[0],
                         reye_pose=pose_righteye[0],
                         left_hand_pose=pose_lefthand[0],
                         right_hand_pose=pose_righthand[0])
    elif model_type == 'smpl':
        raise NotImplementedError('SMPL model not implemented yet.')

