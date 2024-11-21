import torch.utils
import torch.utils.tensorboard
from src.params import Hparams
from src.data_utils.dataset_dataloader import get_train_dataloader
from src.models.unet.unet import UNetModel
from src.models.diffusion.DDPM import VanillaDiffusion
import torch
import os, json
from tqdm import tqdm
import numpy as np


def nested_map(struct, map_fn):
    """This is for transfering into cuda device"""
    if isinstance(struct, tuple):
        return tuple(nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class Learner:

    def __init__(self, hparams: Hparams):
        self.hparams = hparams

        self.output_dir = hparams.output_dir
        self.log_dir = f"{hparams.output_dir}/logs"
        self.checkpoint_dir = f"{hparams.output_dir}/chkpts"

        self.unet = UNetModel(
            in_channels=hparams.in_channels,
            out_channels=hparams.out_channels,
            channels=hparams.channels,
            n_res_blocks=hparams.n_res_blocks,
            attention_levels=hparams.attention_levels,
            channel_multipliers=hparams.channel_multipliers,
            n_heads=hparams.n_heads,
            tf_layers=hparams.tf_layers,
            d_cond=hparams.d_cond,
        ).to(hparams.device)

        self.diffusion = VanillaDiffusion(
            self.unet, hparams.n_steps, hparams.linear_start, hparams.linear_end
        ).to(hparams.device)

        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=hparams.lr_decay_step,
            gamma=hparams.lr_decay_factor,
        )

        self.step = 0
        self.epoch = 0
        self.running_loss_per_50_step = []
        self.running_loss_per_2000_step = []

        self.train_dataloader = get_train_dataloader(hparams)
        self.best_loss = 1e10

        # restore if directory exists
        if os.path.exists(self.output_dir):
            self.restore_from_checkpoint()
        else:
            os.makedirs(self.output_dir)
            os.makedirs(self.log_dir)
            os.makedirs(self.checkpoint_dir)
            with open(f"{self.output_dir}/params.json", "w") as params_file:
                json.dump(self.hparams.to_dict(), params_file)

        print(json.dumps(self.hparams.to_dict(), sort_keys=True, indent=4))

        self.summary_writer = torch.utils.tensorboard.writer.SummaryWriter(
            log_dir=self.log_dir
        )

    def restore_from_checkpoint(self, fname="weights"):
        try:
            fpath = f"{self.checkpoint_dir}/{fname}.pt"
            checkpoint = torch.load(fpath)
            self.load_state_dict(checkpoint)
            print(f"Restored from checkpoint {fpath} --> {fname}-{self.epoch}.pt!")
            return True
        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch...")
            return False

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.diffusion.load_state_dict(state_dict["diffusion"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def state_dict(self):
        model_state = self.diffusion.state_dict()
        return {
            "step": self.step,
            "epoch": self.epoch,
            "diffusion": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            "optimizer": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.optimizer.state_dict().items()
            },
        }

    def _link_checkpoint(self, save_name, link_fpath):
        if os.path.islink(link_fpath):
            os.unlink(link_fpath)
        os.symlink(save_name, link_fpath)

    def save_to_checkpoint(self, fname="weights", is_best=False):
        save_name = f"{fname}-{self.step}.pt"
        save_fpath = f"{self.checkpoint_dir}/{save_name}"
        link_best_fpath = f"{self.checkpoint_dir}/{fname}_best.pt"
        link_fpath = f"{self.checkpoint_dir}/{fname}.pt"
        torch.save(self.state_dict(), save_fpath)
        self._link_checkpoint(save_name, link_fpath)
        if is_best:
            self._link_checkpoint(save_name, link_best_fpath)

    def train(self):

        num_paras = sum(
            p.numel() for p in self.diffusion.parameters() if p.requires_grad
        )
        print(f"Number of parameters: {num_paras}")

        self.running_loss_per_50_step = []
        self.running_loss_per_2000_step = []

        for epoch in range(self.epoch, self.hparams.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            self.scheduler.step()

    def train_epoch(self):
        self.diffusion.train()

        for batch in tqdm(self.train_dataloader):
            batch_images, batch_labels = batch
            batch_images = batch_images.to(self.hparams.device)
            batch_labels = batch_labels.to(self.hparams.device)

            loss = self.diffusion.loss(batch_images, batch_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.running_loss_per_50_step.append(loss.item())
            self.running_loss_per_2000_step.append(loss.item())
            if self.step % 50 == 0:
                loss_per_50 = np.mean(self.running_loss_per_50_step)
                self.running_loss_per_50_step = []
                self.summary_writer.add_scalar("train/loss", loss_per_50, self.step)

                if self.step % 2000 == 0:
                    loss_per_2000 = np.mean(self.running_loss_per_2000_step)
                    is_best = False
                    if loss_per_2000 < self.best_loss:
                        self.best_loss = loss_per_2000
                        is_best = True
                    self.save_to_checkpoint(is_best=is_best)

            self.step += 1
        return
