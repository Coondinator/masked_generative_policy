from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops
import os
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder

from diffusion_policy_3d.model.mgt.act_vq import ActVQ
from diffusion_policy_3d.model.mgt_utils.config_long import vq_args_parser, trans_args_parser
from diffusion_policy_3d.model.mgt.transformer import ActTransformer
import diffusion_policy_3d.model.mgt_utils.losses as losses
from diffusion_policy_3d.model.mgt_utils.utils_model import generate_src_mask, gumbel_sample, cosine_schedule
from pathlib import Path
import random
import numpy as np


class MGT(BasePolicy):
    def __init__(self,
                 shape_meta: dict,
                 # noise_scheduler: DDPMScheduler,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 # # VQ parameters
                 # vq_num_code: int = 512,
                 # vq_code_dim: int = 512,
                 # vq_output_dim: int = 512,
                 # vq_commit_weight: float = 0.25,
                 # # Transformer parameters
                 # trans_embed_dim: int = 512,
                 # trans_num_layers: int = 6,
                 # trans_num_heads: int = 8,
                 # Observation encoder
                 load_vq=False,
                 encoder_output_dim=256,
                 crop_shape=None,
                 use_pc_color=False,
                 pointnet_type="pointnet",
                 pointcloud_encoder_cfg=None,
                 # parameters passed to step
                 **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                 img_crop_shape=crop_shape,
                                 out_channel=encoder_output_dim,
                                 pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                 use_pc_color=use_pc_color,
                                 pointnet_type=pointnet_type,
                                 )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        self.obs_encoder = obs_encoder

        self.args_vq = vq_args_parser()
        self.args_trans = trans_args_parser()
        self.vq_model = self.build_vq(self.args_vq)
        self.trans_model = self.build_trans(self.args_trans)

        if load_vq==True:
            try:
                self.load_vq_checkpoint(device='cuda' if torch.cuda.is_available() else 'cpu')
            except FileNotFoundError as e:
                print(f"Warning: {str(e)}, training from scratch")

        self.losses = losses

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        print_params(self)

    def build_vq(self, args_vq):
        # args = vq_args_parser()
        torch.manual_seed(args_vq.seed)
        torch.cuda.manual_seed(args_vq.seed)
        torch.cuda.manual_seed_all(args_vq.seed)
        np.random.seed(args_vq.seed)
        random.seed(args_vq.seed)

        vq_model = ActVQ(args_vq,  ## use args to define different parameters in different quantizers
                         args_vq.nb_code,
                         args_vq.code_dim,
                         args_vq.output_emb_width,
                         args_vq.down_t,
                         args_vq.stride_t,
                         args_vq.width,
                         args_vq.depth,
                         args_vq.dilation_growth_rate,
                         args_vq.vq_act,
                         args_vq.vq_norm)
        return vq_model

    def build_trans(self, args_trans):
        # args = trans_args_parser()
        torch.manual_seed(args_trans.seed)
        torch.cuda.manual_seed(args_trans.seed)
        torch.cuda.manual_seed_all(args_trans.seed)
        np.random.seed(args_trans.seed)
        random.seed(args_trans.seed)

        vq_model = self.vq_model
        trans = ActTransformer(vqvae=vq_model,
                               num_vq=args_trans.nb_code,
                               embed_dim=args_trans.embed_dim_gpt,
                               comb_state_dim=args_trans.comb_state_dim,
                               #  pc_dim=args_trans.pc_dim,
                               cond_length=args_trans.cond_length,
                               block_size=args_trans.block_size,
                               num_layers=args_trans.num_layers,
                               num_local_layer=args_trans.num_local_layer,
                               n_head=args_trans.n_head_gpt,
                               drop_out_rate=args_trans.drop_out_rate,
                               fc_rate=args_trans.ff_rate)
        return trans

    def load_vq_checkpoint(self, nb_iter=None, device='cpu'):
        vq_out_dir = os.path.join(self.args_vq.out_dir, f'vq/h128')
        ckpt_path = os.path.join(vq_out_dir, 'hand-insert_180000.pth')



        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

        checkpoint = torch.load(ckpt_path, map_location=device)
        # Load weights into model
        state_dict = checkpoint['net']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('vq_model.'):
                new_k = k[len('vq_model.'):]  # Strip 'vq_model.' prefix
                new_state_dict[new_k] = v
            # else:
            #     print(f"Warning: Key {k} not found in model")
        self.vq_model.load_state_dict(new_state_dict)
        print(vq_out_dir)
        print(f"Loaded checkpoint from {ckpt_path}")
        return

    def vq_eval(self):
        self.vq_model.eval()
        for parmeter in self.vq_model.parameters():
            parmeter.requires_grad = False
        return

    def vq_train(self):
        self.vq_model.train()
        for parmeter in self.vq_model.parameters():
            parmeter.requires_grad = True
        return

    def trans_eval(self):
        self.trans_model.eval()
        for parameter in self.trans_model.parameters():
            parameter.requires_grad = False
        return

    def trans_train(self):
        self.trans_model.train()
        for parameter in self.trans_model.parameters():
            parameter.requires_grad = True
        return

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           condition_data_pc=None, condition_mask_pc=None,
                           local_cond=None, global_cond=None,
                           generator=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(sample=trajectory,
                                 timestep=t,
                                 local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def vq_encode(self, x):
        """
        Encode the input data using the VQ model.
        Args:
            x: Input data to be encoded.
        Returns:
            Encoded data.
        """
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0
        x_d = self.vq_model.encode(x)
        return x_d

    def predict_MGT_action(self, batch):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if 'action' in batch:
            action = batch['action'].to(self.device)
            # print('action',action[0])
            action = self.normalizer['action'].normalize(action)
            denormed_gt = self.normalizer['action'].unnormalize(action)
            target_token = self.vq_model.encode(action)  # new code
            batch_size = action.shape[0]  # new code
            m_tokens = self.regression_token_process(target_token)  # new code
            m_tokens_len = torch.full((batch_size,), 2, dtype=m_tokens.dtype).to(m_tokens.device)  # new code
            # m_tokens_len = batch['m_tokens'].to(self.device)
        else:
            m_tokens = batch['m_tokens'].to(self.device)
            m_tokens_len = batch['m_tokens_len'].to(self.device)

        batch['obs']['point_cloud'] = batch['obs']['point_cloud'][:, :4, ...]  # only use first 5 frames
        batch['obs']['agent_pos'] = batch['obs']['agent_pos'][:, :4, ...]

        target = m_tokens.int()
        target = target.cuda()

        obs_dict = {
            'point_cloud': batch['obs']['point_cloud'],
            'agent_pos': batch['obs']['agent_pos']
        }
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']

        this_nobs = dict_apply(nobs, lambda x: x[:, :4, ...].reshape(-1, *x.shape[2:]))
        # this_nobs = dict_apply(nobs, lambda x: x[:,:2,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)

        batch_size, max_len = target.shape[:2]
        nobs_features = nobs_features.reshape(batch_size, 4, -1)

        src_mask = generate_src_mask(
            max_len, m_tokens_len + 1
        )

        with torch.no_grad():
            sampled_tokens = self.trans_model.fast_sample(
                # first_tokens=first_tokens,
                src_mask=src_mask,
                # pc = nobs['point_cloud'],
                comb_state=nobs_features,
                m_length=None,
                step=2,
                gt=target
            )

        decoded_tokens = sampled_tokens[:, 0:2]  # (B, 3)
        decoded_actions = self.vq_model.decode(decoded_tokens)  # (128,12,4)
        # print('mgt',decoded_actions.shape)
        decoded_target = target[:, 0:2]  # (B, 3)
        # print('decoded_target',decoded_target)
        target = self.vq_model.decode(decoded_target)  # (128,12,4)
        # print('mgt',target.shape)
        norm_decoded_actions = decoded_actions

        #####
        norm_decoded_actions = self.normalizer['action'].unnormalize(norm_decoded_actions)
        #####

        return {
            'action_pred': norm_decoded_actions,
            'sampled_tokens': sampled_tokens,
        }

    def predict_MGT_full_action(self, batch, token_length=52):
        if 'action' in batch:
            action = batch['action'].to(self.device)
            # print('action',action[0])
            action = self.normalizer['action'].normalize(action)
            denormed_gt = self.normalizer['action'].unnormalize(action)
            target_token = self.vq_model.encode(action)  # new code
            batch_size = action.shape[0]  # new code
            # m_tokens = self.regression_token_process(target_token)
            m_tokens = target_token[:, :token_length]
            m_tokens_len = torch.full((batch_size,), token_length, dtype=m_tokens.dtype).to(m_tokens.device)  # new code
            # m_tokens_len = batch['m_tokens'].to(self.device)
        else:
            m_tokens = batch['m_tokens'].to(self.device)
            m_tokens_len = batch['m_tokens_len'].to(self.device)

        batch['obs']['point_cloud'] = batch['obs']['point_cloud'][:, :5, ...]  # only use first 5 frames
        batch['obs']['agent_pos'] = batch['obs']['agent_pos'][:, :5, ...]

        target = m_tokens.int()
        target = target.cuda()

        obs_dict = {
            'point_cloud': batch['obs']['point_cloud'],
            'agent_pos': batch['obs']['agent_pos']
        }
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, lambda x: x[:, :5, ...].reshape(-1, *x.shape[2:]))
        # this_nobs = dict_apply(nobs, lambda x: x[:,:2,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        batch_size, max_len = target.shape[:2]
        nobs_features = nobs_features.reshape(batch_size, 5, -1)
        src_mask = generate_src_mask(
            max_len, m_tokens_len + 1
        )
        m_length = torch.full((batch_size,), 208, dtype=torch.long, device=self.device)
        with torch.no_grad():
            sampled_tokens = self.trans_model.fast_sample(
                # first_tokens=first_tokens,
                src_mask=src_mask,
                # pc = nobs['point_cloud'],
                comb_state=nobs_features,
                m_length=208,
                step=10,
                gt=target
            )
        decoded_tokens = sampled_tokens[:, 0:token_length]  # (B, 3)
        norm_decoded_actions = self.vq_model.decode(decoded_tokens)  # (128,12,4)
        norm_decoded_actions = self.normalizer['action'].unnormalize(norm_decoded_actions)

        return {
            'action_pred': norm_decoded_actions,
            'sampled_tokens': sampled_tokens,
            # 'action_gt': norm_target
        }

    def predict_MGT_re(self, batch, tokens, act_time, refine_step, act_length, scores, token_length=51):
        if 'action' in batch:
            action = batch['action'].to(self.device)
            # print('action',action[0])
            action = self.normalizer['action'].normalize(action)
            denormed_gt = self.normalizer['action'].unnormalize(action)

            target_token = self.vq_model.encode(action)  # new code
            batch_size = action.shape[0]  # new code
            m_tokens = target_token[:, :token_length]
            m_tokens_len = torch.full((batch_size,), token_length, dtype=m_tokens.dtype).to(m_tokens.device)  # new code
        else:
            m_tokens = batch['m_tokens'].to(self.device)
            m_tokens_len = batch['m_tokens_len'].to(self.device)

        # print('obs shape:', batch['obs']['point_cloud'].shape)
        batch['obs']['point_cloud'] = batch['obs']['point_cloud'][:, :4, ...]  # only use first 5 frames
        batch['obs']['agent_pos'] = batch['obs']['agent_pos'][:, :4, ...]

        target = m_tokens.int()
        target = target.cuda()

        obs_dict = {
            'point_cloud': batch['obs']['point_cloud'],
            'agent_pos': batch['obs']['agent_pos']
        }
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, lambda x: x[:, :5, ...].reshape(-1, *x.shape[2:]))
        # this_nobs = dict_apply(nobs, lambda x: x[:,:2,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        batch_size, max_len = target.shape[:2]
        nobs_features = nobs_features.reshape(batch_size, 4, -1)
        src_mask = generate_src_mask(
            max_len, m_tokens_len + 1
        )

        with torch.no_grad():
            sampled_tokens, sampled_scores = self.trans_model.regressive_sample_re(
                tokens=tokens,
                src_mask=src_mask,
                act_t=act_time,
                # pc = nobs['point_cloud'],
                comb_state=nobs_features,
                m_length=act_length,
                scores=scores,
                step=refine_step
            )

        decoded_tokens = sampled_tokens[:, 0:token_length]  # (B, 3)
        norm_decoded_actions = self.vq_model.decode(decoded_tokens)  # (128,12,4)
        norm_decoded_actions = self.normalizer['action'].unnormalize(norm_decoded_actions)
        return {
            'action_pred': norm_decoded_actions,
            'sampled_tokens': sampled_tokens,
        }, sampled_scores

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_vq_loss(self, batch):
        # Process observations
        # obs_features = self.obs_encoder(batch['obs'])
        Loss = self.losses.ReConsLoss(recons_loss=self.args_vq.recons_loss, pos_dim=[0, 1, 2], rot_state=False)

        actions = self.normalizer['action'].normalize(batch['action'])
        pred_actions, loss_commit, perplexity = self.vq_model(actions)
        # Calculate reconstruction loss

        ######
        pred_actions = self.normalizer['action'].unnormalize(pred_actions)
        ######
        loss_action = Loss(
            pred_actions,
            batch['action']
        )

        # Total loss
        total_loss = loss_action + self.args_vq.commit * loss_commit

        return total_loss, {
            'loss_recon': loss_action.item(),
            'loss_commit': loss_commit.item(),
            'perplexity': perplexity.item()
        }

    @staticmethod
    def get_acc(cls_pred, target, mask):
        cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
        target_all = torch.masked_select(target, mask)
        probs = torch.softmax(cls_pred, dim=-1)
        _, cls_pred_index = torch.max(probs, dim=-1)
        right_num = (cls_pred_index == target_all).sum()
        return right_num * 100 / mask.sum()

    def regression_token_process(self, token):

        token_length = torch.tensor(2).int()
        batch_size = token.shape[0]

        # else:
        #     token_length = torch.randint(3,  full_token_length, (1,)).view([])
        pad_length = 50 - token_length
        pad_tokens = torch.full((batch_size, pad_length,), self.vq_model.nb_code + 1, dtype=token.dtype).to(
            token.device)

        padded_tokens = torch.cat([token, pad_tokens], dim=1)
        padded_tokens[:, token_length] = self.vq_model.nb_code

        return padded_tokens

    def compute_trans_loss(self, batch):
        # m_tokens = batch['m_tokens'].to(self.device)
        action = batch['action'].to(self.device)
        action = self.normalizer['action'].normalize(action)
        target_token = self.vq_model.encode(action)  # new code
        batch_size = action.shape[0]  # new code
        m_tokens = self.regression_token_process(target_token)  # new code
        m_tokens_len = torch.full((batch_size,), 2, dtype=m_tokens.dtype).to(m_tokens.device)  # new code

        batch['obs']['point_cloud'] = batch['obs']['point_cloud'][:, :4, ...]  # only use first 5 frames
        batch['obs']['agent_pos'] = batch['obs']['agent_pos'][:, :4, ...]

        nobs = self.normalizer.normalize(batch['obs'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

            # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)

        target = m_tokens.int()

        batch_size, max_len = target.shape[:2]
        nobs_features = nobs_features.reshape(batch_size, 4, -1)
        mask = torch.bernoulli(
            self.args_trans.pkeep * torch.ones(target.shape, device=target.device))  # random (0,1) mask
        seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)  # bool mask for the action length

        mask = torch.logical_or(mask, ~seq_mask_no_end).int()
        r_indices = torch.randint_like(target, self.args_trans.nb_code)
        input_indices = mask * target + (1 - mask) * r_indices

        mask_id = self.args_trans.nb_code + 2  # a special token
        rand_mask_probs = torch.zeros(batch_size, device=m_tokens_len.device).float().uniform_(0.5, 1)
        num_token_masked = (m_tokens_len * rand_mask_probs).round().clamp(min=1)
        seq_mask = generate_src_mask(max_len, m_tokens_len + 1)

        batch_randperm = torch.rand((batch_size, max_len), device=target.device) - seq_mask_no_end.int()
        batch_randperm = batch_randperm.argsort(dim=-1)
        # print('batch_randperm:', batch_randperm)
        mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        masked_input_indices = torch.where(mask_token, mask_id, input_indices)

        cls_pred = self.trans_model(masked_input_indices, src_mask=seq_mask, comb_state=nobs_features)[:, 0:]
        weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
        cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
        target_seq_masked = target[seq_mask_no_end]
        weight_seq_masked = weights[seq_mask_no_end]

        loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked.long(), reduction='none')
        loss_cls = (loss_cls * weight_seq_masked).sum()

        probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
        _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
        target_seq_masked = torch.masked_select(target, seq_mask_no_end)
        right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()
        no_mask_token = ~mask_token * seq_mask_no_end
        acc_masked = self.get_acc(cls_pred, target, mask_token)
        acc_no_masked = self.get_acc(cls_pred, target, no_mask_token)
        acc_overall = right_seq_masked * 100 / seq_mask_no_end.sum()
        return loss_cls, {
            'loss_recon': loss_cls.item(),
            'acc_masked': acc_masked.item(),
            'acc_no_masked': acc_no_masked.item(),
            'acc_overall': acc_overall.item()}


    def compute_trans_long_loss(self, batch, rand=True, full_ratio=0.7, obs_len=5):
        # m_tokens = batch['m_tokens'].to(self.device)
        action = batch['action'].to(self.device)
        action = self.normalizer['action'].normalize(action)
        target_token = self.vq_model.encode(action)  # new code
        batch_size = action.shape[0]  # new code
        # m_tokens = self.regression_token_process(target_token) # new code
        # m_tokens_len = torch.full((batch_size,), 2, dtype=m_tokens.dtype).to(m_tokens.device)# new code

        if rand:
            B, T = target_token.shape
            # print(f"Data: {target_token}")
            full_len = torch.full((B,), T, dtype=torch.long, device=self.device)  # full length for each sample
            rand_lens = torch.randint(4, T, (B,), device=self.device)  # random lengths between 4 and T
            m_tokens_len = torch.where(torch.rand(B, device=self.device) < full_ratio, full_len, rand_lens)

            # calculate start index
            max_start = T - m_tokens_len
            start_idxs = torch.floor(torch.rand(B, device=self.device) * (max_start + 1)).long()


            target_idx = start_idxs.unsqueeze(1) + torch.arange(T, device=self.device).unsqueeze(0)  # [B, T]
            mask = torch.arange(T, device=self.device).unsqueeze(0) < m_tokens_len.unsqueeze(1)  # [B, T]

            target_idx = target_idx.clamp(max=T - 1)

            cropped_x = torch.zeros_like(target_token)
            gathered = torch.gather(target_token, 1, target_idx)
            cropped_x[mask] = gathered[mask]

            m_tokens = torch.zeros_like(target_token)
            m_tokens[mask] = target_token[mask]

            obs_horizon = torch.arange(obs_len, device=self.device).unsqueeze(0)
            gather_idx = 4 * start_idxs.unsqueeze(1) + obs_horizon
            batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, obs_len)
            batch['obs']['point_cloud'] = batch['obs']['point_cloud'].to(self.device)
            batch['obs']['agent_pos'] = batch['obs']['agent_pos'].to(self.device)
            batch['obs']['point_cloud'] = batch['obs']['point_cloud'][
                batch_idx, gather_idx, ...]  # only use first obs_len frames
            batch['obs']['agent_pos'] = batch['obs']['agent_pos'][batch_idx, gather_idx, ...]
        else:
            batch['obs']['point_cloud'] = batch['obs']['point_cloud'].to(self.device)
            batch['obs']['agent_pos'] = batch['obs']['agent_pos'].to(self.device)
            batch['obs']['point_cloud'] = batch['obs']['point_cloud'][:, :5, ...]  # only use first 5 frames
            batch['obs']['agent_pos'] = batch['obs']['agent_pos'][:, :5, ...]
            # nobs = self.normalizer.normalize(batch['obs'])
            m_tokens = target_token
            m_tokens_len = torch.full((batch_size,), m_tokens.shape[1], dtype=m_tokens.dtype).to(
                m_tokens.device)  # new code
        nobs = self.normalizer.normalize(batch['obs'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

            # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)

        target = m_tokens.int()
        batch_size, max_len = target.shape[:2]
        nobs_features = nobs_features.reshape(batch_size, 5, -1)

        mask = torch.bernoulli(
            self.args_trans.pkeep * torch.ones(target.shape, device=target.device))  # random (0,1) mask

        seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)  # bool mask for the action length

        mask = torch.logical_or(mask, ~seq_mask_no_end).int()

        r_indices = torch.randint_like(target, self.args_trans.nb_code)
        input_indices = mask * target + (1 - mask) * r_indices

        mask_id = self.args_trans.nb_code + 2  # a special token
        rand_mask_probs = torch.zeros(batch_size, device=m_tokens_len.device).float().uniform_(0.5, 1)
        num_token_masked = (m_tokens_len * rand_mask_probs).round().clamp(min=1)
        # the number of tokens to force-mask in that sample
        seq_mask = generate_src_mask(max_len, m_tokens_len + 1)
        # it “extends” the valid region by one token, thereby including the end token as a valid position.
        # seq_mask = generate_src_mask(max_len, m_tokens_len)
        batch_randperm = torch.rand((batch_size, max_len), device=target.device) - seq_mask_no_end.int()
        batch_randperm = batch_randperm.argsort(dim=-1)
        mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        masked_input_indices = torch.where(mask_token, mask_id, input_indices)


        cls_pred = self.trans_model(masked_input_indices, src_mask=seq_mask, comb_state=nobs_features)[:, 0:]
        weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
        cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
        target_seq_masked = target[seq_mask_no_end]
        weight_seq_masked = weights[seq_mask_no_end]

        loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked.long(), reduction='none')
        loss_cls = (loss_cls * weight_seq_masked).sum()

        probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
        _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
        target_seq_masked = torch.masked_select(target, seq_mask_no_end)
        right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()
        no_mask_token = ~mask_token * seq_mask_no_end
        acc_masked = self.get_acc(cls_pred, target, mask_token)
        acc_no_masked = self.get_acc(cls_pred, target, no_mask_token)
        acc_overall = right_seq_masked * 100 / seq_mask_no_end.sum()
        return loss_cls, {
            'loss_recon': loss_cls.item(),
            'acc_masked': acc_masked.item(),
            'acc_no_masked': acc_no_masked.item(),
            'acc_overall': acc_overall.item()}


    def compute_regress_loss(self, batch, rand=True, full_ratio=0.6, obs_len=4):
        # m_tokens = batch['m_tokens'].to(self.device)
        action = batch['action'].to(self.device)

        action = self.normalizer['action'].normalize(action)
        target_token = self.vq_model.encode(action)  # new code
        batch_size = action.shape[0]  # new code
        if rand:
            B, T = target_token.shape
            full_len = torch.full((B,), T, dtype=torch.long, device=self.device)  # full length for each sample
            rand_lens = torch.randint(8, T, (B,), device=self.device)  # random lengths between 4 and T
            m_tokens_len = torch.where(torch.rand(B, device=self.device) < full_ratio, full_len, rand_lens) - 1 #preserve full_ratio as full length
            max_start = T - m_tokens_len
            token_offset = self.random_offset(m_tokens_len, T)  # token offset

            start_idxs = torch.floor(torch.rand(B, device=self.device) * (max_start-1)).long() + 1  #

            target_idx = start_idxs.unsqueeze(1) + torch.arange(T, device=self.device).unsqueeze(0)  # [B, T]
            mask = torch.arange(T, device=self.device).unsqueeze(0) < m_tokens_len.unsqueeze(1)
            target_idx = target_idx.clamp(max=T - 1)
            cropped_x = torch.full_like(target_token, self.args_trans.nb_code + 1)
            gathered = torch.gather(target_token, 1, target_idx)
            cropped_x[mask] = gathered[mask]
            m_tokens = cropped_x

            obs_horizon = torch.arange(obs_len, device=self.device).unsqueeze(0)
            gather_idx = 4 * (start_idxs + token_offset).unsqueeze(1) - obs_horizon

            batch_idx = torch.arange(B, device=self.device)
            batch_idx_obs = batch_idx.unsqueeze(1).expand(-1, obs_len)
            batch['obs']['point_cloud'] = batch['obs']['point_cloud'].to(self.device)
            batch['obs']['agent_pos'] = batch['obs']['agent_pos'].to(self.device)
            batch['obs']['point_cloud'] = batch['obs']['point_cloud'][batch_idx_obs, gather_idx, ...]
            batch['obs']['agent_pos'] = batch['obs']['agent_pos'][batch_idx_obs, gather_idx, ...]
        else:
            batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, obs_len)

            batch['obs']['point_cloud'] = batch['obs']['point_cloud'].to(self.device)
            batch['obs']['agent_pos'] = batch['obs']['agent_pos'].to(self.device)
            batch['obs']['point_cloud'] = batch['obs']['point_cloud'][:, 1:5, ...]  # only use first 5 frames
            batch['obs']['agent_pos'] = batch['obs']['agent_pos'][:, 1:5, ...]
            m_tokens = target_token
            m_tokens_len = torch.full((batch_size,), m_tokens.shape[1], dtype=m_tokens.dtype).to(m_tokens.device)

        nobs = self.normalizer.normalize(batch['obs'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        target = m_tokens.int()

        batch_size, max_len = target.shape[:2]
        nobs_features = nobs_features.reshape(batch_size, 4, -1)

        mask = torch.bernoulli(
            self.args_trans.pkeep * torch.ones(target.shape, device=target.device))  # random (0,1) mask

        seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)  # bool mask for the action length
        mask = torch.logical_or(mask, ~seq_mask_no_end).int()

        range = torch.arange(T, device=self.device).unsqueeze(0)
        token_offset_expanded = token_offset.unsqueeze(1)
        time_mask = (range < token_offset_expanded)  # shape: [batch, T] 保留token——offset之前的数据

        mask = mask.masked_fill(time_mask, 1) # 将 time_mask 中的 True 设置为 1（强制保留）
        r_indices = torch.randint_like(target, self.args_trans.nb_code)  # random index terbulance
        input_indices = mask * target + (1 - mask) * r_indices
        # print('input_indices', input_indices[0])
        mask_id = self.args_trans.nb_code + 2  # a special token
        rand_mask_probs = torch.zeros(batch_size, device=m_tokens_len.device).float().uniform_(0.5, 1)
        num_token_masked = ((m_tokens_len-token_offset) * rand_mask_probs).round().clamp(min=1)
        # the number of tokens to force-mask in that sample
        seq_mask = generate_src_mask(max_len, m_tokens_len + 1)

        batch_randperm = torch.rand((batch_size, max_len), device=target.device) - seq_mask_no_end.int()
        batch_randperm = batch_randperm.argsort(dim=-1)

        mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        mask_token = mask_token.masked_fill(time_mask, False)  # mask token

        masked_input_indices = torch.where(mask_token, mask_id, input_indices)
        masked_input_indices[batch_idx, m_tokens_len] = self.args_trans.nb_code # add end token

        cls_pred = self.trans_model(masked_input_indices, src_mask=seq_mask, comb_state=nobs_features)[:, 0:]
        weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
        cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
        target_seq_masked = target[seq_mask_no_end]
        weight_seq_masked = weights[seq_mask_no_end]

        loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked.long(), reduction='none')
        loss_cls = (loss_cls * weight_seq_masked).sum()

        probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
        _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
        target_seq_masked = torch.masked_select(target, seq_mask_no_end)
        right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()
        no_mask_token = ~mask_token * seq_mask_no_end
        acc_masked = self.get_acc(cls_pred, target, mask_token)
        acc_no_masked = self.get_acc(cls_pred, target, no_mask_token)
        acc_overall = right_seq_masked * 100 / seq_mask_no_end.sum()
        return loss_cls, {
            'loss_recon': loss_cls.item(),
            'acc_masked': acc_masked.item(),
            'acc_no_masked': acc_no_masked.item(),
            'acc_overall': acc_overall.item()}

    def random_offset(self, m_token_length, T):
        B = m_token_length.shape[0]
        offset_zero = torch.zeros((B,), dtype=torch.long, device=self.device)  # full length for each sample
        token_offset = torch.floor(torch.rand(B, device=self.device) * (m_token_length - 2)).long()  # start token idx [1 -> max idx]
        real_offset = torch.where(torch.rand(B, device=self.device) < 0.3, offset_zero, token_offset)
        return real_offset