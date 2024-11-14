from src.learner import Learner
from src.params import Hparams
import argparse

# batch_size=8,
#         num_workers=4,
#         in_channels=1,
#         out_channels=1,
#         channels=32,
#         n_res_blocks=1,
#         attention_levels=[4],
#         channel_multipliers=[1, 2, 2, 2, 2],
#         n_heads=4,
#         d_cond=128,
#         tf_layers=1,
#         n_steps=100,
#         linear_start=1e-4,
#         linear_end=2e-2,
#         lr=1e-4,
#         weight_decay=1e-4,
#         lr_decay_step=5,
#         lr_decay_factor=0.5,
#         output_dir="outputs",
#         max_epoch=30,


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--n_res_blocks", type=int, default=1)

    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--linear_start", type=float, default=1e-4)
    parser.add_argument("--linear_end", type=float, default=2e-2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_decay_step", type=int, default=2)
    parser.add_argument("--lr_decay_factor", type=float, default=0.3)

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_epoch", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hparams = Hparams(**vars(args))
    learner = Learner(hparams)
    learner.train()
