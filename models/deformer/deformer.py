import torch.nn as nn

from models.deformer.rigid import get_rigid_deform
from models.deformer.non_rigid import get_non_rigid_deform

class Deformer(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.rigid = get_rigid_deform(cfg.rigid, metadata)
        self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata)

    def forward(self, gaussians, camera, iteration, compute_loss=True):
        loss_reg = {}
        
        deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)
        
        deformed_gaussians, pts_W = self.rigid(deformed_gaussians, iteration, camera)

        gaussians.set_xyz_J(pts_W)


        # get the joint weights of each points
        # gaussians.set_xyz_J(self.rigid.get_xyz_J(deformed_gaussians))
        if (iteration in [100, 2100,  6100,10100]):
            # import ipdb; ipdb.set_trace()
            gaussians.save_ply(f"point_cloud/gaussian_{iteration}.ply")
            deformed_gaussians.save_ply(f"point_cloud/deformed_gaussian_{iteration}.ply")
            gaussians.save_weights(f"point_cloud/gaussian_weights_{iteration}.txt")


        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg

def get_deformer(cfg, metadata):
    return Deformer(cfg, metadata)