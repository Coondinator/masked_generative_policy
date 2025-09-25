if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import zarr
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.mgt_utils.utils_model import initial_optim

import torch.optim as optim
from torch.utils.data import RandomSampler
from diffusion_policy_3d.mgt_policy.mgt import MGT
from pathlib import Path

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = MGT(shape_meta=cfg.policy.shape_meta,
            # noise_scheduler: DDPMScheduler,
                         horizon=cfg.policy.horizon,
                         n_action_steps=cfg.policy.horizon,
                         n_obs_steps=cfg.policy.horizon,
                         encoder_output_dim=cfg.policy.encoder_output_dim,
                         load_vq=True,
                         crop_shape=cfg.policy.crop_shape,
                         use_pc_color=cfg.policy.use_pc_color,
                         pointnet_type=cfg.policy.pointnet_type,
                         pointcloud_encoder_cfg=cfg.policy.pointcloud_encoder_cfg)
        
        self.optimizer = initial_optim(self.model.args_trans.decay_option, self.model.args_trans.lr, self.model.args_trans.weight_decay, self.model.trans_model, self.model.args_trans.optimizer)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.model.args_trans.lr_scheduler, gamma=self.model.args_trans.gamma)

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.task_name = cfg.task.name

    @staticmethod
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    def run(self):
        cfg = copy.deepcopy(self.cfg)


        # configure dataset
        dataset: BaseDataset
        cfg.task.trans_dataset._target_ = 'diffusion_policy_3d.dataset.metaworld_vq_dataset.MetaworldDataset'
        cfg.task.trans_dataset.pad_before = 4
        cfg.task.trans_dataset.pad_after = 4
        cfg.dataloader.shuffle = False
        cfg.dataloader.batch_size = 16
        cfg.val_dataloader.batch_size = 16
        dataset = hydra.utils.instantiate(cfg.task.trans_dataset)
        sampler = RandomSampler(dataset, replacement=True, num_samples=100)
        train_dataloader = DataLoader(dataset, sampler=sampler, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        train_dataloader_iter = self.cycle(train_dataloader)
        test_dataloader_iter = self.cycle(val_dataloader)
        # configure validation dataset
        self.model.set_normalizer(normalizer)
        print('save iter:', self.model.args_trans.save_iter)
        env_runner_MGT: BaseRunner
        env_runner_MGT = hydra.utils.instantiate(
            cfg.task.env_runner_MGT,
            output_dir=self.output_dir)
        # if env_runner is not None:
        #     assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint("-----------------------------", "yellow")
        use_wandb = True

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)
        self.model.vq_eval()

        # training loop
        for nb_iter in tqdm(range(1, self.model.args_trans.total_iter + 1), position=0, leave=True):
            self.model.trans_train()
            step_log = dict()
            # ========= train for this epoch ==========
            batch = next(train_dataloader_iter)
            # if nb_iter % self.model.args_trans.eval_rand_iter == 0:
                # print('real action', batch['action'][0])
            loss_cls, loss_dict = self.model.compute_regress_loss(batch)
            self.optimizer.zero_grad()
            loss_cls.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            if nb_iter % self.model.args_trans.print_iter == 0:
                print(f'Iter {nb_iter} : Loss. {loss_cls:.5f}, ACC. {loss_dict["acc_overall"]:.4f}',
                    f'ACC_masked. {loss_dict["acc_masked"]:.4f}', f'ACC_no_masked. {loss_dict["acc_no_masked"]:.4f}')

            # ========= eval for this epoch ==========
            policy = self.model
            policy.eval()
          # run validation
            if nb_iter % self.model.args_trans.eval_rand_iter == 0:
                self.model.trans_eval()
                test_loss_total = 0.0
                test_acc_total = 0.0
                test_mask_acc_total = 0.0
                test_no_mask_acc_total = 0.0
                num_batches = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        loss_cls, loss_dict = self.model.compute_regress_loss(val_batch)
                
                        test_loss_total += loss_dict['loss_recon']
                        test_acc_total += loss_dict['acc_overall']
                        test_mask_acc_total += loss_dict['acc_masked']
                        test_no_mask_acc_total += loss_dict['acc_no_masked']
                        num_batches +=1
                
                test_mask_acc_mean = test_mask_acc_total/ num_batches
                test_no_mask_acc_mean = test_no_mask_acc_total/ num_batches
                test_loss_mean = test_loss_total/ num_batches
                test_acc_mean = test_acc_total/ num_batches
                
                print(f'Iter {nb_iter} : Val_Rand_Loss. {test_loss_mean:.5f}, Val_Rand_ACC. {test_acc_mean:.4f}',
              f'ACC_masked. {test_mask_acc_mean:.4f}', f'ACC_no_masked. {test_no_mask_acc_mean:.4f}')

                result = self.model.predict_MGT_re(batch, token_length=52)
                gt_action = batch['action'].to(device)
                pred_action = result['action_pred']
                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                step_log['train_action_mse_error'] = mse.item()
                print(f'Iter {nb_iter} : train_action_mse_error. {step_log["train_action_mse_error"]:.5f}')
                
                if use_wandb:
                    wandb.log({
                        "Val/Iteration": nb_iter,
                        # "Val/Loss_mse": test_loss_mse_mean,
                        # "Val/Total_trans_Loss": test_loss_mean,
                        "Val/Loss": test_loss_mean,
                        "Val/ACC": test_acc_mean,
                        "Val/ACC_masked": test_mask_acc_mean,
                        "Val/ACC_no_masked": test_no_mask_acc_mean,
                        "Val/train_action_mse_error": step_log["train_action_mse_error"],
                    }, step=nb_iter)
                
                del batch
                # del obs_dict
                del gt_action
                del result
                del pred_action
                del mse
                
            if nb_iter % self.model.args_trans.eval_env == 0:
                runner_log = env_runner_MGT.run_regress(self.model, action_step=16, refine_step=1)

                cprint(f"---------------- Eval Results --------------", 'magenta')
                for key, value in runner_log.items():
                    if isinstance(value, float):
                        cprint(f"{key}: {value:.4f}", 'magenta')
            # checkpoint
            if nb_iter % self.model.args_trans.save_iter == 0:

                self.save_checkpoint(path=None,nb_iter=nb_iter)


    def eval(self):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)
        RUN_VALIDATION = False  # reduce time cost
        # # resume training
        # if cfg.training.resume:
        #     lastest_ckpt_path = self.get_checkpoint_path()
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        cfg.task.trans_dataset._target_ = 'diffusion_policy_3d.dataset.metaworld_vq_dataset.MetaworldDataset'
        cfg.task.trans_dataset.pad_before = 4
        cfg.task.trans_dataset.pad_after = 4
        cfg.dataloader.shuffle = False
        cfg.dataloader.batch_size = 16
        cfg.val_dataloader.batch_size = 16
        dataset = hydra.utils.instantiate(cfg.task.trans_dataset)
        normalizer = dataset.get_normalizer()
        # configure validation dataset
        self.model.set_normalizer(normalizer)
        print('save iter:', self.model.args_trans.save_iter)
        env_runner_MGT: BaseRunner
        env_runner_MGT = hydra.utils.instantiate(
            cfg.task.env_runner_MGT,
            output_dir=self.output_dir)

        cfg = copy.deepcopy(self.cfg)
        checkpoint_path = "checkpoints/basketball/checkpoint_iter_40000.pth"  # or your specific path
        if Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        else:
            print("No checkpoint found, starting from scratch")

        cfg = copy.deepcopy(self.cfg)
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.trans_dataset)

        policy = self.model
        policy.eval()
        policy.cuda()

        runner_log = env_runner_MGT.run_regress(self.model, action_step=16, refine_step=1)
        # runner_log = env_runner_MGT.run_test(policy,first_obs)
        
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def test_env(self, cfg, data_dir):
        output_dir0 = 'output/test_env/'
        output_dir1 = 'output/test_env/'
        device = self.model.device
        visual_data = zarr.open(data_dir, mode='r')
        episode_ends = visual_data['meta']['episode_ends']
        episode_ends = episode_ends[:] if hasattr(episode_ends, '__getitem__') else episode_ends

        val_action = visual_data['data/action'][0: episode_ends[0]].astype(np.float32)
    
        env_runner_MGT0: BaseRunner
        env_runner_MGT0 = hydra.utils.instantiate(cfg.task.env_runner_MGT, output_dir0)
        
        for i in range(5):
            _ = env_runner_MGT0.test_run(val_action, save_dir=output_dir0, save_video=True, type='test_env_ep0_'+str(i))  # real_train_action
        
        env_runner_MGT1: BaseRunner
        env_runner_MGT1 = hydra.utils.instantiate(cfg.task.env_runner_MGT, output_dir1)

        for i in range(5):
            _ = env_runner_MGT1.test_run(val_action, save_dir=output_dir1, save_video=True, type='test_env_ep1_'+str(i))

        return

    def save_checkpoint(self, path=None,nb_iter=None):
        if path is None:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            path = checkpoint_dir / f"checkpoint_iter_{nb_iter}.pth"
        print(f'Trying to save trans ckpt to {path}')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")

    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_original_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload

    def load_checkpoint(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint {path} not found")
        
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path})")

    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    # workspace.run()
    workspace.eval()

if __name__ == "__main__":
    main()
