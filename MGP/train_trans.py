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
        # self.model: DP3 = hydra.utils.instantiate(cfg.policy)
        self.model = MGT(shape_meta=cfg.policy.shape_meta,
            # noise_scheduler: DDPMScheduler,
            horizon=cfg.policy.horizon, 
            n_action_steps=cfg.policy.horizon, 
            n_obs_steps=cfg.policy.horizon,
            encoder_output_dim=cfg.policy.encoder_output_dim,
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
        RUN_VALIDATION = False # reduce time cost     
        # # resume training
        # if cfg.training.resume:
        #     lastest_ckpt_path = self.get_checkpoint_path()
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.trans_dataset)
        # dataset = hydra.utils.instantiate(
        #     cfg.task.trans_dataset, 
        #     phase='train'  # Explicit training phase
        # )
        # val_dataset = hydra.utils.instantiate(
        #     cfg.task.trans_dataset,
        #     phase='val'    # Explicit validation phase
        # )
        # assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        train_dataloader_iter = self.cycle(train_dataloader)
        test_dataloader_iter = self.cycle(val_dataloader)
        # configure validation dataset


        self.model.set_normalizer(normalizer)

        # configure env
        # env_runner: BaseRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        env_runner_MGT: BaseRunner
        env_runner_MGT = hydra.utils.instantiate(
            cfg.task.env_runner_MGT,
            output_dir=self.output_dir)
        # if env_runner is not None:
        #     assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        use_wandb = True   
        if use_wandb:  
            self.wandb_run = wandb.init(
                project="diffusion_policy_MGT",
                name='trans',
                group=cfg.logging.group
            )
        else:
            self.wandb_run = None  
        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # checkpoint_path = "checkpoints/checkpoint_iter_80000.pth"  # or your specific path
        # self.load_checkpoint(checkpoint_path)

        # device transfer
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
            loss_cls, loss_dict = self.model.compute_trans_loss(batch)
            self.optimizer.zero_grad()
            loss_cls.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            if nb_iter % self.model.args_trans.print_iter == 0:
                print(f'Iter {nb_iter} : Loss. {loss_cls:.5f}, ACC. {loss_dict["acc_overall"]:.4f}',
                    f'ACC_masked. {loss_dict["acc_masked"]:.4f}', f'ACC_no_masked. {loss_dict["acc_no_masked"]:.4f}')
                # print(f'Iter {nb_iter} : Total_trans_Loss. {total_loss:.5f}, Loss_recons. {loss_dict["loss_recon"]:.5f}, Loss_mse. {loss_dict["loss_mse"]:.5f}, ACC. {loss_dict["acc_overall"]:.4f}',
                #     f'ACC_masked. {loss_dict["acc_masked"]:.4f}', f'ACC_no_masked. {loss_dict["acc_no_masked"]:.4f}')
                if use_wandb:
                    wandb.log({
                        "Train/Iteration": nb_iter,
                        # "Train/Total_trans_Loss": total_loss,
                        "Train/Loss": loss_cls,
                        # "Train/Loss_mse": loss_dict["loss_mse"],
                        "Train/ACC": loss_dict["acc_overall"],
                        "Train/ACC_masked": loss_dict["acc_masked"],
                        "Train/ACC_no_masked": loss_dict["acc_no_masked"],
                    }, step=nb_iter)

            # ========= eval for this epoch ==========
            # policy = self.model
            # policy.eval()          
          # run validation
            if nb_iter % self.model.args_trans.eval_rand_iter == 0:
                # policy = self.model
                # policy.eval()
                self.model.trans_eval()
                test_loss_total = 0.0
                # test_loss_mse = 0.0
                # test_loss_recons = 0.0
                test_acc_total =  0.0
                test_mask_acc_total =  0.0
                test_no_mask_acc_total =  0.0
                num_batches = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        loss_cls, loss_dict = self.model.compute_trans_loss(val_batch)
                
                        test_loss_total += loss_dict['loss_recon']
                        # test_loss_recons += loss_dict['loss_recon']
                        # test_loss_mse += loss_dict['loss_mse']
                        test_acc_total += loss_dict['acc_overall']
                        test_mask_acc_total += loss_dict['acc_masked']
                        test_no_mask_acc_total += loss_dict['acc_no_masked']
                        num_batches +=1
                
                test_mask_acc_mean = test_mask_acc_total/ num_batches
                test_no_mask_acc_mean = test_no_mask_acc_total/ num_batches
                test_loss_mean = test_loss_total/ num_batches
                # test_loss_mse_mean = test_loss_mse/ num_batches
                # test_loss_recons_mean = test_loss_recons/ num_batches
                test_acc_mean = test_acc_total/ num_batches
                
                print(f'Iter {nb_iter} : Val_Rand_Loss. {test_loss_mean:.5f}, Val_Rand_ACC. {test_acc_mean:.4f}',
              f'ACC_masked. {test_mask_acc_mean:.4f}', f'ACC_no_masked. {test_no_mask_acc_mean:.4f}')
            #     print(f'Iter {nb_iter} : Val_Rand_Loss. {test_loss_mean:.5f}, Val_mse_Loss. {test_loss_mse_mean:.5f}, Val_recons_Loss. {test_loss_recons_mean:.5f}, Val_Rand_ACC. {test_acc_mean:.4f}',
            #   f'ACC_masked. {test_mask_acc_mean:.4f}', f'ACC_no_masked. {test_no_mask_acc_mean:.4f}')
                
                # sample trajectory from training set, and evaluate difference
                # batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                # obs_dict = batch['obs']
                # gt_action = batch['action']
                # batch = next(train_dataloader_iter) 
                result = self.model.predict_MGT_action(batch)
                # gt_action = result['action_gt']
                gt_action = batch['action'].to(device)
                # print(f'Iter {nb_iter} : Val_Rand_GT_Action. {gt_action}')
                # print(gt_action.shape) 50 12 4
                pred_action = result['action_pred']
                # print(f'Iter {nb_iter} : Val_Rand_Pred_Action. {pred_action}')
                # print(pred_action.shape) 50 12 4
                # print('pred_action', pred_action[0])
                # print('gt_action', gt_action[0])
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
                
                if nb_iter % self.model.args_trans.eval_env==0:
                    runner_log = env_runner_MGT.run(self.model)
        
                    cprint(f"---------------- Eval Results --------------", 'magenta')
                    for key, value in runner_log.items():
                        if isinstance(value, float):
                            cprint(f"{key}: {value:.4f}", 'magenta')
                    if use_wandb:
                        wandb.log({
                            "Eval/Iteration": nb_iter,
                            **runner_log
                        }, step=nb_iter)
                # policy.train()
            # checkpoint
            if nb_iter % self.model.args_trans.save_iter == 0:
                # checkpointing
                
                self.save_checkpoint(path=None,nb_iter=nb_iter)
                
                # # sanitize metric names
                # metric_dict = dict()
                # for key, value in step_log.items():
                #     new_key = key.replace('/', '_')
                #     metric_dict[new_key] = value
                
                # # We can't copy the last checkpoint here
                # # since save_checkpoint uses threads.
                # # therefore at this point the file might have been empty!
                # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                # if topk_ckpt_path is not None:
                #     self.save_checkpoint(path=topk_ckpt_path)
     
        if use_wandb:
            wandb.finish()


    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        checkpoint_path = "checkpoints/checkpoint_iter_150000.pth"  # or your specific path
        if Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        else:
            print("No checkpoint found, starting from scratch")
        # lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        # if lastest_ckpt_path.is_file():
        #     cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
        #     self.load_checkpoint(path=lastest_ckpt_path)
        
        cfg = copy.deepcopy(self.cfg)
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.trans_dataset)
        # first_sample = dataset[0]
        # first_obs = {
        #     'point_cloud': first_sample['obs']['point_cloud'][0:5],
        #     'agent_pos': first_sample['obs']['agent_pos'][0:5]
        # }
        # configure env

        # self.visual_trans(cfg, cfg.task.trans_dataset.zarr_path)
        self.test_env(cfg, cfg.task.trans_dataset.zarr_path)

        env_runner_MGT: BaseRunner
        env_runner_MGT = hydra.utils.instantiate(
            cfg.task.env_runner_MGT,
            output_dir=self.output_dir)
        assert isinstance(env_runner_MGT, BaseRunner)
        policy = self.model
        policy.eval()
        policy.cuda()

        runner_log = env_runner_MGT.run(policy)
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
    

    def visual_trans(self, cfg, data_dir):
        output_dir = 'mgt_output/trans_visual/'    
        self.model.vq_eval()
        self.model.trans_eval()
        self.model.cuda()
        device = self.model.device
        visual_data = zarr.open(data_dir, mode='r')
        episode_ends = visual_data['meta']['episode_ends']
        episode_ends = episode_ends[:] if hasattr(episode_ends, '__getitem__') else episode_ends

        val_action = visual_data['data/action'][0: episode_ends[0]].astype(np.float32)
        val_agent_pos = torch.from_numpy(visual_data['data/state'][0: episode_ends[0]].astype(np.float32)).to(device='cuda')
        val_point_cloud = torch.from_numpy(visual_data['data/point_cloud'][0: episode_ends[0]].astype(np.float32)).to(device='cuda')
        
        train_action = visual_data['data/action'][episode_ends[0]: episode_ends[1]].astype(np.float32)
        train_agent_pos = torch.from_numpy(visual_data['data/state'][episode_ends[0]: episode_ends[1]].astype(np.float32)).to(device='cuda')
        train_point_cloud = torch.from_numpy(visual_data['data/point_cloud'][episode_ends[0]: episode_ends[1]].astype(np.float32)).to(device='cuda')
        
        episode_length = train_action.shape[0]
        trunck_num = episode_length // 5

        env_runner_MGT: BaseRunner
        env_runner_MGT = hydra.utils.instantiate(cfg.task.env_runner_MGT, output_dir)
        
        _ = env_runner_MGT.test_run(train_action, save_dir=output_dir, save_video=True,  type='trans_train_real')
        _ = env_runner_MGT.test_run(val_action, save_dir=output_dir, save_video=True,  type='trans_val_real')  # real_train_action
        
        print('train_agent_pos shape', train_agent_pos.shape)
        train_agent_pos = torch.cat([train_agent_pos[0:1].repeat(3, 1), train_agent_pos], dim=0)
        print('pad train_agent_pos shape', train_agent_pos.shape)
        print('train_point_cloud shape', train_point_cloud.shape)
        print('pad train_point_cloud shape', train_point_cloud[0:1].repeat(3, 1, 1).shape)
        train_point_cloud = torch.cat([train_point_cloud[0:1].repeat(3, 1, 1), train_point_cloud], dim=0)

        val_agent_pos = torch.cat([val_agent_pos[0:1].repeat(3, 1), val_agent_pos], dim=0)
        val_point_cloud = torch.cat([val_point_cloud[0:1].repeat(3, 1, 1), val_point_cloud], dim=0)

        
        batch_size = 1
        print('trunk_num', trunck_num)

        for i in range(trunck_num):
            cond_index = i * 5
            train_agent_pos_clip = train_agent_pos[cond_index: cond_index+4].unsqueeze(0)
            train_point_cloud_clip = train_point_cloud[cond_index: cond_index+4].unsqueeze(0)

            val_agent_pos_clip = val_agent_pos[cond_index: cond_index+4].unsqueeze(0)
            val_point_cloud_clip = val_point_cloud[cond_index: cond_index+4].unsqueeze(0)

            dummy_m_tokens_len = torch.tensor([50] * batch_size, device=device)
            dummy_m_tokens = torch.zeros((batch_size, 50), dtype=torch.long, device=device)
            
            train_batch = {
                'm_tokens': dummy_m_tokens,
                # 'pc': pc,
                # 'state': state,
                'm_tokens_len': dummy_m_tokens_len
            } 
            train_batch['obs'] = {
                'point_cloud': train_point_cloud_clip,
                'agent_pos': train_agent_pos_clip
            }

            train_action_dict = self.model.predict_MGT_action(train_batch)
            np_action_dict = dict_apply(train_action_dict,
                                        lambda x: x.detach().to('cpu').numpy())
            predict_train_clip = np_action_dict['action_pred'][:, 3:, ...].squeeze(0)


            val_batch = {
                'm_tokens': dummy_m_tokens,
                # 'pc': pc,
                # 'state': state,
                'm_tokens_len': dummy_m_tokens_len
            } 
            val_batch['obs'] = {
                'point_cloud': val_point_cloud_clip,
                'agent_pos': val_agent_pos_clip
            }

            val_action_dict = self.model.predict_MGT_action(val_batch)
            np_action_dict = dict_apply(val_action_dict,
                                        lambda x: x.detach().to('cpu').numpy())
            predict_val_clip = np_action_dict['action_pred'][:, 3:, ...].squeeze(0)


            if i == 0:
                pred_train_action_seq = predict_train_clip
                pred_val_action_seq = predict_val_clip
            else:
                pred_train_action_seq = np.concatenate((pred_train_action_seq, predict_train_clip), axis=0)
                pred_val_action_seq = np.concatenate((pred_val_action_seq, predict_val_clip), axis=0)

        runner_log = env_runner_MGT.test_run(pred_val_action_seq, save_dir=output_dir, save_video=True, type='trans_val_pred') # real_val_action
        runner_log = env_runner_MGT.test_run(pred_train_action_seq, save_dir=output_dir, save_video=True, type='trans_train_pred')  # pred_val_action
        
        return
 


    def test_env(self, cfg, data_dir):
        output_dir0 = 'mgt_output/test_env/'
        output_dir1 = 'mgt_output/test_env/'     
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



    def save_original_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def save_checkpoint(self, path=None,nb_iter=None):
        if path is None:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            path = checkpoint_dir / f"checkpoint_iter_{nb_iter}.pth"
        
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
    workspace.run()

if __name__ == "__main__":
    main()
