from src.data_utils.dataset_dataloader import get_train_dataloader
from src.params import Hparams
from src.models.unet.unet import UNetModel
from src.models.diffusion.DDPM import VanillaDiffusion
import torch
from tqdm import tqdm

if __name__ == "__main__":

    hparams = Hparams(dataset_dir="subclass12", batch_size=8, num_workers=4)
    train_dataloader = get_train_dataloader(hparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNetModel(
        in_channels=1,
        out_channels=1,
        channels=32,
        n_res_blocks=1,
        attention_levels=[4],
        channel_multipliers=[1, 2, 2, 2, 2],
        n_heads=4,
        d_cond=128,
    ).to("cuda")

    diffusion = VanillaDiffusion(
        unet, hparams.n_steps, hparams.linear_start, hparams.linear_end
    ).to("cuda")
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    num_paras = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_paras}")

    for epoch in range(3):
        print(f"Epoch: {epoch}")
        running_loss = []
        for x in tqdm(train_dataloader):
            img, label = x
            img = img.to("cuda")
            label = label.to("cuda")
            loss = diffusion.loss(img, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        print(f"Epoch {epoch} loss: {sum(running_loss) / len(running_loss)}")
