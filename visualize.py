import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, Evaluator, PSEvaluator
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss

import hydra
from omegaconf import OmegaConf
import wandb
import lpips
import matplotlib.pyplot as plt

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import trimesh
import igl

def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value

def visualizing(config):
    model = config.model
    dataset = config.dataset
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    gaussians = GaussianModel(model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    # scene.visualize()
    metadata = scene.metadata
    skinning_weights = metadata["skinning_weights"]
    cano_mesh = metadata["cano_mesh"]
    smpl_verts = metadata["smpl_verts"]
    hand_mesh = metadata['cano_hand_mesh']
    # print("cano_mesh: ", cano_mesh)
    # print("cano_hand_mesh: ", hand_mesh)
    pts_skinning, pts_W_hand = hand_sampling(metadata)
    save_handsamples(pts_skinning, pts_W_hand)

    

def save_skinning(metadata):

    smpl_verts = metadata["smpl_verts"]
    skinning_weights = metadata["skinning_weights"]
    model_type = 'smplx' # hard-coded model type, used for skinning weights visualization
    if model_type == 'smplx':
        num_joints = 55 # assuming on smplx model
        joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
        joint_colors = joint_colors[:,:3]

    points_skinning = torch.tensor(smpl_verts).cpu()
    pred_weights = torch.tensor(skinning_weights).cpu()
    print("pred_weights: ", pred_weights.shape)
    max_joint_indices = torch.argmax(pred_weights, dim=1)
    point_colors = joint_colors[max_joint_indices]
    if model_type == 'smplx':
        with open(f'point_cloud_init.txt', 'w') as f:
            for point, color in zip(points_skinning, point_colors):
                # Write XYZRGB data to the file
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

    print("XYZRGB file created successfully.")

def hand_labelling(metadata):
    skinning_weights = metadata["skinning_weights"]
    smpl_verts = metadata["smpl_verts"]
    points_skinning = torch.tensor(smpl_verts).cpu()
    # labeling the face in trimesh
    pred_weights = torch.tensor(skinning_weights).cpu()
    max_joint_indices = torch.argmax(pred_weights, dim=1)
    faces = metadata["faces"]
    faces = torch.tensor(faces).cpu()
    # capture the points related to hand joints
    left_wrist = 20
    right_wrist = 21
    finger_start_idx = 25
    finger_end_idx = 54
    hand_vertices = torch.nonzero((max_joint_indices >= finger_start_idx) & (max_joint_indices <= finger_end_idx) | (max_joint_indices == left_wrist) | (max_joint_indices == right_wrist)).squeeze()

    #labeling face containing the hand vertices
    vertex_matches = torch.eq(faces.unsqueeze(2), hand_vertices.view(1, 1, -1))
    num_matches_per_vertex = torch.sum(vertex_matches, dim=2)
    num_occurrences_per_row = torch.sum(num_matches_per_vertex, dim=1)
    hand_mesh_idx = torch.nonzero(num_occurrences_per_row >= 3).squeeze()
    print("cano_hand_mesh: ", metadata["cano_hand_mesh"])
    # test: set all hand vertices to red
    with open(f'point_cloud_hand.txt', 'w') as f:
        for i, point in zip(range(points_skinning.size(dim = 0)),points_skinning):
            if(i in hand_vertices):
                color = torch.tensor([1.0, 0.0, 0.0])
            else:
                color = torch.tensor([0.0, 1.0, 0.0])
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

def hand_sampling(metadata):
    smpl_verts = metadata["smpl_verts"]
    cano_hand_mesh = metadata["cano_hand_mesh"]
    skinning_weights = metadata["skinning_weights"]
    points_skinning, face_idx = cano_hand_mesh.sample(1024, return_index=True)
    hand_faces = cano_hand_mesh.faces
    hand_verts = cano_hand_mesh.vertices
    points_skinning = points_skinning.view(np.ndarray).astype(np.float32)
    hand_verts = hand_verts.view(np.ndarray).astype(np.float32)
    hand2cano = metadata['hand2cano_dict']
    bary_coords = igl.barycentric_coordinates_tri(
        points_skinning,
        hand_verts[hand_faces[face_idx, 0], :],
        hand_verts[hand_faces[face_idx, 1], :],
        hand_verts[hand_faces[face_idx, 2], :],
    )
    vert_ids_hand = hand_faces[face_idx, ...]
    vert_ids_cano = hand2cano[vert_ids_hand]
    pts_W_hand = (skinning_weights[vert_ids_cano] * bary_coords[..., None]).sum(axis=1)
    points_skinning = torch.from_numpy(points_skinning).cuda()
    pts_W_hand = torch.from_numpy(pts_W_hand).cuda()
    return points_skinning, pts_W_hand

def save_handsamples(points_skinning, pts_W_hand):
    num_joints = 55 # assuming on smplx model
    joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
    joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
    joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
    joint_colors = joint_colors[:,:3]
    points = points_skinning.detach().cpu()
    weights = pts_W_hand.detach().cpu()
    maxjoint_idx = torch.argmax(weights, dim=1)
    point_colors = joint_colors[maxjoint_idx]
    with open(f'hand_sampling.txt', 'w') as f:
                with open(f'weights_verification.txt', 'w') as f:
                    for point, color in zip(points, point_colors):
                        # Write XYZRGB data to the file
                        f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

    

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False) # allow adding new values to config

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)
    # config.checkpoint_iterations.append(config.opt.iterations)

    # # set wandb logger
    # wandb_name = config.name
    # wandb.init(
    #     # mode="disabled",
    #     mode="disabled" if config.wandb_disable else None,
    #     name=wandb_name,
    #     # project='gaussian-splatting-avatar',
    #     project='3dgs',
    #     entity='digitalhumans',
    #     dir=config.exp_dir,
    #     config=OmegaConf.to_container(config, resolve=True),
    #     settings=wandb.Settings(start_method='fork'),
    # )



    print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    visualizing(config)

    # All done
    print("\nVisualizing completed.")


if __name__ == "__main__":
    main()