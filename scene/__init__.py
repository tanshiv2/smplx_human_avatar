#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from models import GaussianConverter
from scene.gaussian_model import GaussianModel
from dataset import load_dataset
import matplotlib.pyplot as plt
import numpy as np 


class BaseScene:
    def __init__(self, cfg, save_dir : str):
        self.cfg = cfg
        self.save_dir = save_dir
        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError



class Scene(BaseScene):

    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, save_dir : str):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians = gaussians

        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError

        self.cameras_extent = self.metadata['cameras_extent']

        self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(), spatial_lr_scale=self.cameras_extent)

        self.converter = GaussianConverter(cfg, self.metadata).cuda()

        self.model_type = 'smpl' # hard-coded model type, used for skinning weights visualization
        self.save_skinning = False
        if self.model_type == 'smplx':
            num_joints = 55 # assuming on smplx model
            self.joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
            self.joint_colors = np.vstack((self.joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
            self.joint_colors = np.vstack((self.joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
            self.joint_colors = self.joint_colors[:,:3]
        elif self.model_type == 'smpl':
            num_joints = 25
            self.joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
            self.joint_colors = np.vstack((self.joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 5))))
            self.joint_colors = self.joint_colors[:,:3]


    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    def optimize(self, iteration):
        gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        if iteration >= gaussians_delay:
            self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize()

        if self.save_skinning:
            # save the xyzrgb point cloud for visualization if required
            interval = 100
            if iteration % interval == 0:
                self.vzy_skinning(iteration)

    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)

    def get_skinning_loss(self):
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        pts_skinning = loss_reg.get('pts_skinning', torch.tensor(0.).cuda())
        sampled_weights = loss_reg.get('sampled_weights', torch.tensor(0.).cuda())
        pred_weights = loss_reg.get('pred_weights', torch.tensor(0.).cuda())

        return loss_skinning, pts_skinning, sampled_weights, pred_weights


    def vzy_weights(self, pts_skinning, sampled_weights, pred_weights, iteration):
        joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
        joint_colors = joint_colors[:,:3]
        points = pts_skinning.detach().cpu()
        pred = pred_weights.detach().cpu()
        sampled = sampled_weights.detach().cpu()
        maxjoint_pred_idx = torch.argmax(pred, dim=1)
        maxjoint_sampled_idx = torch.argmax(sampled, dim=1)
        pred_colors = joint_colors[maxjoint_pred_idx]
        sampled_colors = joint_colors[maxjoint_sampled_idx]
        with open(f'pred_weights_{iteration}.txt', 'w') as f:
            for point, color in zip(points, pred_colors):
                # Write XYZRGB data to the file
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        with open(f'sampled_weights_{iteration}.txt', 'w') as f:
            for point, color in zip(points, sampled_colors):
                # Write XYZRGB data to the file
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        print("XYZRGB file created successfully.")

    def save(self, iteration):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((self.gaussians.capture(),
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def load_checkpoint(self, path):
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        self.converter.load_state_dict(converter_sd)
        # self.converter.optimizer.load_state_dict(converter_opt_sd)
        # self.converter.scheduler.load_state_dict(converter_scd_sd)

    def visualize(self):
        # pcd = self.test_dataset.readPointCloud()
        # np.savez('point_cloud_data.npz', points=pcd.points, colors=pcd.colors, normals=pcd.normals)
        # visualize the initial gaussian models
        print("Visualizing the initial gaussian models")

    def vzy_skinning(self, iteration):
        # skinning_path = os.path.join(self.save_dir, "skinning/iteration_{}".format(iteration))
        # os.makedirs(skinning_path, exist_ok=True)
        points_skinning = self.converter.deformer.rigid.points_skinning.detach().cpu()
        pred_weights = self.converter.deformer.rigid.pred_weights.detach().cpu()
        print("pred_weights: ", pred_weights.shape)
        max_joint_indices = torch.argmax(pred_weights, dim=1)
        point_colors = self.joint_colors[max_joint_indices]
        if self.model_type == 'smplx':
            with open(f'point_cloud_{iteration}.txt', 'w') as f:
                for point, color in zip(points_skinning, point_colors):
                    # Write XYZRGB data to the file
                    f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        elif self.model_type == 'smpl':
            with open(f'point_cloud_{iteration}_zju.txt', 'w') as f:
                for point, color in zip(points_skinning, point_colors):
                    # Write XYZRGB data to the file
                    f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

        print("XYZRGB file created successfully.")


        
class Scene_H(BaseScene):

    # gaussians : GaussianModel

    def __init__(self, cfg, gaussians_list, save_dir : str):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians_list = gaussians_list
        # Gaussians List
        # 0: body
        # 1: left wrist, 20
        # 2: right wrist, 21
        # 3: left index 1, 22
        # 4: left index 2
        # 5: left index 3
        # 6: left middle 1
        # 7: left middle 2
        # 8: left middle 3
        # 9: left pinky 1
        # 10: left pinky 2
        # 11: left pinky 3
        # 12: left ring 1
        # 13: left ring 2
        # 14: left ring 3
        # 15: left thumb 1
        # 16: left thumb 2
        # 17: left thumb 3
        # 18: right index 1
        # 19: right index 2
        # 20: right index 3
        # 21: right middle 1
        # 22: right middle 2
        # 23: right middle 3
        # 24: right pinky 1
        # 25: right pinky 2
        # 26: right pinky 3
        # 27: right ring 1
        # 28: right ring 2
        # 29: right ring 3
        # 30: right thumb 1
        # 31: right thumb 2
        # 32: right thumb 3

        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError

        self.cameras_extent = self.metadata['cameras_extent']
        self.gaussians_list[0].create_from_pcd(self.test_dataset.readPointCloud_body(), spatial_lr_scale=self.cameras_extent)
        for i in range(0, 32):
            self.gaussians_list[i+1].create_from_pcd(self.test_dataset.readPointCloud_hand(i), spatial_lr_scale=self.cameras_extent)



        self.converter = GaussianConverter(cfg, self.metadata).cuda()

        self.model_type = 'smpl' # hard-coded model type, used for skinning weights visualization
        self.save_skinning = False
        if self.model_type == 'smplx':
            num_joints = 55 # assuming on smplx model
            self.joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
            self.joint_colors = np.vstack((self.joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
            self.joint_colors = np.vstack((self.joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
            self.joint_colors = self.joint_colors[:,:3]
        elif self.model_type == 'smpl':
            num_joints = 25
            self.joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
            self.joint_colors = np.vstack((self.joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 5))))
            self.joint_colors = self.joint_colors[:,:3]


    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    def optimize(self, iteration):
        gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        if iteration >= gaussians_delay:
            self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize()

        if self.save_skinning:
            # save the xyzrgb point cloud for visualization if required
            interval = 100
            if iteration % interval == 0:
                self.vzy_skinning(iteration)

    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        # return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)
        converter_list = []
        for i in range(34):
            converter_list.append(self.converter(self.gaussians_list[i], viewpoint_camera, iteration, compute_loss))
        return converter_list