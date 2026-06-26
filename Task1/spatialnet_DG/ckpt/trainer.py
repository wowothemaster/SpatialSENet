import os
import torch
import random
import shutil
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from pesq import pesq
from joblib import Parallel, delayed
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from utils.distributed_utils import reduce_value
from utils.stft import STFT
from utils.norm import Norm

class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self.STFT = STFT(**self.config['FFT'])
        self.norm = Norm('frequency', online=False)

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        # training config
        config['DDP']['world_size'] = args.world_size
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']

        self.exp_path = self.trainer_config['exp_path']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        self.code_path = os.path.join(self.exp_path, 'codes')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)
        
        # save the config and codes
        if self.rank == 0:
            data = OmegaConf.create(config)
            OmegaConf.save(data, os.path.join(self.exp_path, 'config.yaml'))

            shutil.copy2(__file__, self.exp_path)
            for file in Path(__file__).parent.iterdir():
                if file.is_file():
                    shutil.copy2(file, self.code_path)
            shutil.copytree(Path(__file__).parent / 'models', Path(self.code_path) / 'models', dirs_exist_ok=True)
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 0

        self._resume_checkpoint()

        self.log_path = os.path.join(self.exp_path, 'logs')
        # create log file
        self.log_file = open(os.path.join(self.log_path, 'train.log'), 'a')

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, score):
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'model': model_dict}
        
        pesq_str = f"{score:.2f}"
        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch+1).zfill(3)}_pesq{pesq_str}.tar'))

        if score > self.best_score:
            # self.state_dict_best = state_dict.copy()
            self.best_score = score

            best_name = f"best_model.tar"
            best_path = os.path.join(self.checkpoint_path, best_name)
            torch.save(state_dict, best_path)
            print(f"[BEST] New best model saved → {best_path}")
            self._log(f"[Epoch {epoch+1}] BEST updated → PESQ={score:.4f}")

    def _resume_checkpoint(self):
        ckpt_files = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))
        
        if len(ckpt_files) == 0:
            print(f"[INFO] No checkpoint found in {self.checkpoint_path}. Start training from scratch.")
            self.start_epoch = 0
            return
        latest_checkpoint = ckpt_files[-1]
        print(f"[INFO] Resuming from checkpoint: {latest_checkpoint}")

        map_location = self.device
        checkpoint = torch.load(latest_checkpoint, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

    def _log(self, msg):
        if self.rank == 0:     # only main process writes log
            self.log_file.write(msg + '\n')
            self.log_file.flush()

    def _train_epoch(self, epoch):
        total_loss = 0
        if hasattr(self.train_dataloader.dataset, "sample_data_per_epoch"):
            self.train_dataloader.dataset.sample_data_per_epoch()
        self.train_bar = tqdm(self.train_dataloader, ncols=110)

        for step, (clean,noisy,_) in enumerate(self.train_bar, 1):

            clean_f, stft_paras_n = self.STFT.stft(clean)
            noisy_f, stft_paras_n = self.STFT.stft(noisy) # [B, C, F, T]
            B, C, F, T = noisy_f.shape
            noisy_f, (Xr_n, XrMM_n) = self.norm.norm(noisy_f, ref_channel=0)
            noisy_f = noisy_f.permute(0, 2, 3, 1) # [B, F, T, C]
            noisy_f = torch.view_as_real(noisy_f).reshape(B, F, T, -1)
            clean     = clean.to(self.device)
            clean_f   = clean_f.to(self.device)
            noisy_f   = noisy_f.to(self.device)
            
            enhanced_f = self.model(noisy_f)

            enhanced_f = torch.view_as_complex(enhanced_f.float().reshape(B, F, T, -1)) # [B, F, T, C]
            enhanced_f = enhanced_f.unsqueeze(-1).permute(0, 3, 1, 2) # [B, C, F, T]
            enhanced_f = self.norm.inorm(enhanced_f, (Xr_n.to(self.device), XrMM_n.to(self.device)))
            enhanced   = self.STFT.istft(enhanced_f, stft_paras_n)
                
            loss, _ = self.loss_func(enhanced, enhanced_f, clean[:,0,:].unsqueeze(1), clean_f[:,0,:].unsqueeze(1))
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            self.train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch+1, self.epochs, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_bar.postfix = 'train_loss={:.6f}'.format(total_loss / step)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()

            if self.config['scheduler']['update_interval'] == 'step':
                self.scheduler.step()

        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': total_loss / step}, epoch)

            lr = self.optimizer.param_groups[0]['lr']
            self._log(f"[Epoch {epoch+1}] Train   | loss={total_loss / step:.6f} | lr={lr:.6e}")


    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_pesq_score = 0

        self.validation_bar = tqdm(self.validation_dataloader, ncols=123)
        for step, (clean,noisy,_) in enumerate(self.validation_bar, 1):
            clean_f, stft_paras_n = self.STFT.stft(clean)
            noisy_f, stft_paras_n = self.STFT.stft(noisy) # [B, C, F, T]
            B, C, F, T = noisy_f.shape
            noisy_f, (Xr_n, XrMM_n) = self.norm.norm(noisy_f, ref_channel=0)
            noisy_f = noisy_f.permute(0, 2, 3, 1) # [B, F, T, C]
            noisy_f = torch.view_as_real(noisy_f).reshape(B, F, T, -1)
            clean     = clean.to(self.device)
            clean_f   = clean_f.to(self.device)
            noisy_f   = noisy_f.to(self.device)
            
            enhanced_f = self.model(noisy_f)

            if not torch.is_complex(enhanced_f):
                enhanced_f = torch.view_as_complex(enhanced_f.float().reshape(B, F, T, -1))
            enhanced_f = enhanced_f.unsqueeze(-1).permute(0, 3, 1, 2)
            enhanced_f = self.norm.inorm(enhanced_f, (Xr_n.to(self.device), XrMM_n.to(self.device)))
            enhanced   = self.STFT.istft(enhanced_f, stft_paras_n)

            loss, _ = self.loss_func(enhanced, enhanced_f, clean[:,0,:].unsqueeze(1), clean_f[:,0,:].unsqueeze(1))
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            clean = clean.cpu().numpy()
            enhanced = enhanced.detach().cpu().numpy()
            pesq_score_batch = Parallel(n_jobs=1)(
                delayed(pesq)(16000, c, e, 'wb') for c, e in zip(clean[:,0,:], enhanced[:,0,:]))
            pesq_score = torch.tensor(pesq_score_batch, device=self.device).mean()
            if self.world_size > 1:
                pesq_score = reduce_value(pesq_score)
            total_pesq_score += pesq_score
            
            if self.rank == 0 and (epoch==1 or epoch %10 == 0) and step <= 3:
                noisy_path = os.path.join(self.sample_path, 'sample_{}_noisy.wav'.format(step))
                clean_path = os.path.join(self.sample_path, 'sample_{}_clean.wav'.format(step))
                enhanced_path = os.path.join(self.sample_path, 'sample_{}_enh_epoch{}.wav'.format(step, str(epoch).zfill(3)))
                if not os.path.exists(noisy_path):
                    noisy = noisy.cpu().numpy()
                    sf.write(noisy_path, noisy[0].T, samplerate=self.config['samplerate'])
                    sf.write(clean_path, clean[0].T, samplerate=self.config['samplerate'])

                sf.write(enhanced_path, enhanced[0].T, samplerate=self.config['samplerate'])

            self.validation_bar.desc = 'validate[{}/{}][{}]'.format(
                epoch+1, self.epochs, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_bar.postfix = 'valid_loss={:.3f}, pesq={:.4f}'.format(
                total_loss / step, total_pesq_score / step)

        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('val_loss', {'val_loss': total_loss / step, 'pesq': total_pesq_score / step}, epoch)
            self._log(f"[Epoch {epoch+1}] Valid   | loss={total_loss / step:.6f} | pesq={total_pesq_score / step:.4f}")

        return total_loss / step, total_pesq_score / step


    def train(self):
        # self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss, score = self._validation_epoch(epoch)
            
            if self.config['scheduler']['update_interval'] == 'epoch':
                if self.config['scheduler']['use_plateau']:
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, score)

        # if self.rank == 0:
        #     torch.save(self.state_dict_best,
        #             os.path.join(self.checkpoint_path,
        #             'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(3))))

        #     print('------------Training for {} epochs is done!------------'.format(self.epochs))
