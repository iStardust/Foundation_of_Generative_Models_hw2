import torch


class Hparams:
    def __init__(
        self,
        dataset_dir="subclass12",
        batch_size=8,
        num_workers=4,
        in_channels=1,
        out_channels=1,
        channels=32,
        n_res_blocks=1,
        attention_levels=[4],
        channel_multipliers=[2, 2, 2, 2, 2],
        n_heads=4,
        d_cond=128,
        tf_layers=1,
        n_steps=100,
        linear_start=1e-4,
        linear_end=2e-2,
        lr=1e-4,
        weight_decay=1e-4,
        lr_decay_step=2,
        lr_decay_factor=0.3,
        output_dir="outputs",
        max_epoch=8,
    ):
        # dataset and dataloader parameters
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # UNet
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.n_res_blocks = n_res_blocks
        self.attention_levels = attention_levels
        self.channel_multipliers = channel_multipliers
        self.n_heads = n_heads
        self.d_cond = d_cond
        self.tf_layers = tf_layers

        # Diffusion
        self.n_steps = n_steps
        self.linear_start = linear_start
        self.linear_end = linear_end

        # Optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        # Scheduler
        self.lr_decay_step = lr_decay_step
        self.lr_decay_factor = lr_decay_factor

        # Training
        self.output_dir = output_dir
        self.max_epoch = max_epoch

    def to_dict(self):
        return {
            "dataset_dir": self.dataset_dir,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "channels": self.channels,
            "n_res_blocks": self.n_res_blocks,
            "attention_levels": self.attention_levels,
            "channel_multipliers": self.channel_multipliers,
            "n_heads": self.n_heads,
            "d_cond": self.d_cond,
            "tf_layers": self.tf_layers,
            "n_steps": self.n_steps,
            "linear_start": self.linear_start,
            "linear_end": self.linear_end,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "lr_decay_step": self.lr_decay_step,
            "lr_decay_factor": self.lr_decay_factor,
            "output_dir": self.output_dir,
            "max_epoch": self.max_epoch,
        }
