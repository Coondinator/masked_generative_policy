import torch
import torch.nn as nn


class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, pos_dim, rot_state, rot_dim = [3, 4, 5]):
        super(ReConsLoss, self).__init__()

        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss()

        if rot_state == True:
            self.rot_dim = rot_dim

        self.bce_loss = nn.BCELoss()

        self.pos_dim = pos_dim

    def forward(self, action_pred, action_gt):
        loss = self.Loss(action_pred, action_gt)

        return loss #pos_loss + bc_loss

    def forward_joint(self, motion_pred, motion_gt):
        loss = self.Loss(motion_pred[..., 4: (self.nb_joints - 1) * 3 + 4],
                         motion_gt[..., 4: (self.nb_joints - 1) * 3 + 4])
        return loss

