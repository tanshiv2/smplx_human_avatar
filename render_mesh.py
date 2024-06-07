import os
import sys
import torch
import pytorch3d
import numpy as np
from glob import glob
from tqdm import tqdm

from pytorch3d.io import load_ply
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scipy import ndimage

from pytorch3d.transforms import axis_angle_to_matrix

from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    SoftSilhouetteShader,
    SoftPhongShader,
    BlendParams,
)

import matplotlib.pyplot as plt
import cv2

root = "../../data/X_Humans/00036"
gender = "female"
split = "test"
takes = sorted(glob(os.path.join(root, split, "*")))

skinning_weights_path = "./body_models/misc/skinning_weights_all_smplx.npz"
skinning_weights = np.load(skinning_weights_path, allow_pickle=True)[gender]
joint_labels = np.argmax(skinning_weights, axis=1)
joint_labels = torch.tensor(joint_labels, dtype=torch.int64)

colors_left_hand = torch.tensor([[1.0, 0.0, 0.0] for _ in range(len(joint_labels))], dtype=torch.float32)
colors_right_hand = torch.tensor([[0.0, 0.0, 1.0] for _ in range(len(joint_labels))], dtype=torch.float32)
colors_others = torch.tensor([[0.0, 0.0, 0.0] for _ in range(len(joint_labels))], dtype=torch.float32)

left_hand_mask = torch.logical_or(joint_labels == 20, torch.logical_and(joint_labels >= 25, joint_labels <= 39)).unsqueeze(1).repeat(1, 3)
right_hand_mask = torch.logical_or(joint_labels == 21, torch.logical_and(joint_labels >= 40, joint_labels <= 54)).unsqueeze(1).repeat(1, 3)

colors = torch.where(left_hand_mask, colors_left_hand, colors_others)
colors = torch.where(right_hand_mask, colors_right_hand, colors)

# colors = torch.zeros_like(colors)


for take in takes:
    print(take)
    outdir = os.path.join(take, "render", "hand_masks")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(outdir)
    mesh_paths = sorted(glob(os.path.join(take, "SMPLX/*.ply")))
    gt_image_paths = sorted(glob(os.path.join(take, "render/image/*.png")))
    camera_path = os.path.join(take, "render/cameras.npz")

    cameras = np.load(camera_path, allow_pickle=True)
    K = cameras["intrinsic"]
    extrinsics = cameras["extrinsic"]

    focal_length_x = K[0, 0]
    focal_length_y = K[1, 1]
    FovX = focal2fov(focal_length_x, 800)
    FovY = focal2fov(focal_length_y, 1200)

    for index, mesh_path in enumerate(tqdm(mesh_paths)):
        extrinsic = extrinsics[index]
        R = extrinsic[:3, :3]
        T = extrinsic[:3, 3]
        R = R.transpose()

        R = np.expand_dims(R, axis=0)
        T = np.expand_dims(T, axis=0)

        verts, faces = load_ply(mesh_path)
        verts = verts.cuda()
        faces = faces.cuda()

        mesh = Meshes(verts=[verts], faces=[faces])
        vertex_colors = TexturesVertex(verts_features=colors[None, :, :].float()).to("cuda")
        mesh.textures = vertex_colors

        camera = FoVPerspectiveCameras(
            device="cuda",
            R=R,
            T=T,
            fov=FovX,
            degrees=False,
            aspect_ratio=1.0,
        )

        raster_settings = RasterizationSettings(
            image_size=(1200, 800),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )

        lights = AmbientLights(device="cuda")

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device="cuda",
                cameras=camera,
                lights=lights,
            ),
        )

        images = renderer(mesh, cameras=camera, lights=lights)
        image_np = images[0, :, :, :3].cpu().numpy() * 255

        image_np = ndimage.rotate(image_np, 180)
        image_np = cv2.resize(image_np, (540, 540), interpolation=cv2.INTER_LINEAR)

        outpath = os.path.join(outdir, os.path.basename(mesh_path).replace(".ply", ".png"))


        # gt_image = cv2.imread(gt_image_paths[index])
        # gt_image = cv2.resize(gt_image, (540, 540), interpolation=cv2.INTER_LANCZOS4)
        # gt_image_masked = np.where(image_np < 1, gt_image, 0)
        # cv2.imwrite("test.png", gt_image_masked)
        # cv2.imwrite("test_mask.png", image_np)
        # import ipdb; ipdb.set_trace()

        cv2.imwrite(outpath, image_np)



