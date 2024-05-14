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
        
        # get the joint weights of each points
        # gaussians.set_xyz_J(self.rigid.get_xyz_J(deformed_gaussians))
        # if (iteration % 1000 == 0):
        #     import ipdb; ipdb.set_trace()

        deformed_gaussians, pts_W = self.rigid(deformed_gaussians, iteration, camera)

        gaussians.set_xyz_J(pts_W)

        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg

def get_deformer(cfg, metadata):
    return Deformer(cfg, metadata)