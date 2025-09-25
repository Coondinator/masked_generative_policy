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

        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.pos_dim = pos_dim
        # self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4

    def forward(self, action_pred, action_gt):
        loss = self.Loss(action_pred, action_gt)
        # pos_loss = self.Loss(action_pred[..., self.pos_dim], action_gt[..., self.pos_dim])
        # print('pos_loss:', pos_loss.item())
        # print('action_pred:', action_gt[..., -1])
        # bc_loss = self.bce_loss(action_pred[..., -1], action_gt[..., -1])
        # print('bc_loss:', bc_loss.item())
        return loss #pos_loss + bc_loss

    def forward_joint(self, motion_pred, motion_gt):
        loss = self.Loss(motion_pred[..., 4: (self.nb_joints - 1) * 3 + 4],
                         motion_gt[..., 4: (self.nb_joints - 1) * 3 + 4])
        return loss

