import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import glob
import math
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import DiT and its configuration from dit.py in the same folder.
from dit import DiT, DiTConfig

START_CHECKPOINT = "dit_epoch_100.pt"

# ------------------------------------------------------------------
# Define a simple Dataset to load hippo images.
# Since our images are all in one folder (hippo_images) and are PNGs,
# we use glob to list all the files.
# ------------------------------------------------------------------
class HippoDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


# ------------------------------------------------------------------
# Define a helper that builds a linear noise schedule.
# We compute “betas” linearly from a starting value to an ending value,
# then the cumulative product of (1-beta) is used at training.
# ------------------------------------------------------------------
def get_noise_schedule(timesteps, device, s=0.008):
    # Create timesteps (include an extra point for cumulative product)
    steps = timesteps + 1
    t_lin = torch.linspace(0, timesteps, steps, device=device) / timesteps

    # Compute cumulative alphas using the cosine schedule.
    # Here, the cosine squared function (with a small shift s) yields
    # a smooth schedule for the cumulative product ᾱ(t).
    alphas_cumprod = torch.cos(((t_lin + s) / (1 + s)) * (math.pi * 0.5)) ** 2
    # Normalize so that ᾱ(0) = 1.
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Derive betas from the alphas cumulative product:
    # beta_t = 1 - ᾱ(t+1)/ᾱ(t)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Clamp betas for numerical stability.
    betas = torch.clamp(betas, 0, 0.999)
    
    # Now compute alphas for each step and the cumulative product (ᾱ)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    return betas, alphas, alpha_bars

# ------------------------------------------------------------------
# Main training loop:
#
# For each batch, we:
#  1. Sample a random timestep (t) for every image.
#  2. Sample random Gaussian noise.
#  3. Create a noisy image: 
#       x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*noise
#  4. Forward the noisy image (with time t) through the DiT,
#     which is trained to predict the noise.
#  5. Use an MSE loss between the predicted and true noise.
#
# We log training metrics to wandb as well.
# ------------------------------------------------------------------

def train():
    wandb.init(project="dit-ddpm-mnist", config={
        "input_size": 64,        # The input resolution (64x64)
        "patch_size": 8,
        "in_channels": 3,
        "dim": 512,              # Reduced overall width for a small network
        "depth": 4,              # Fewer transformer blocks
        "dim_head": 128,          # Smaller attention heads
        "mlp_mult": 4,
        "time_emb_dim": 128,
        "timesteps": 32,        # Experiment with 100, 200, etc.
        "beta_start": 3e-5,
        "beta_end": 0.02, 
        "lr": 5e-4,
        "batch_size": 64,
        "epochs": 10000,
        "dataset_path": "hippo_images_face"
    })
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # (Define your transforms, dataset, and dataloader here)
    # For example, using your existing HippoDataset:
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda t: t * 2 - 1),  # mapping [0,1] -> [-1,1]
    ])
    
    dataset = HippoDataset(root=config.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=4, drop_last=True)

    # Instantiate the DiT model as before.
    from dit import DiT, DiTConfig
    model_config = DiTConfig(
        input_size=config.input_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        dim=config.dim,
        depth=config.depth,
        dim_head=config.dim_head,
        mlp_mult=config.mlp_mult,
        time_emb_dim=config.time_emb_dim,
    )
    model = DiT(model_config).to(device)
    if START_CHECKPOINT:
        state_dict = torch.load(START_CHECKPOINT)
        model.load_state_dict(state_dict, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Build noise schedule (if needed in training)
    T = config.timesteps
    betas, alphas, alpha_bars = get_noise_schedule(T, device, s=config.beta_start)

    global_step = 0
    # Outer progress bar for epochs:
    outer_pbar = tqdm(range(config.epochs), desc="Training Epochs", unit="epoch")
    for epoch in outer_pbar:
        model.train()
        epoch_loss = 0.0

        for i, x0 in enumerate(dataloader):
            x0 = x0.to(device)  # x0 in range [-1,1]
            batch_size = x0.shape[0]
            
            # Sample random time steps for each example.
            t = torch.randint(0, T, (batch_size,), device=device)
            alpha_bar_t = alpha_bars[t].view(batch_size, 1, 1, 1)
            
            # Sample noise and create the noisy image x_t.
            noise = torch.randn_like(x0)
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
            x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

            t_float = t.float()  # model expects time as float
            pred_noise, fin = model(x_t, t_float)
            loss1 = nn.functional.mse_loss(pred_noise, noise)
            loss2 = nn.functional.mse_loss(fin, x0)

            loss = loss1 + loss2 * 0.6
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            if global_step % 100 == 0:
                wandb.log({"loss": loss.item(), "global_step": global_step, "epoch": epoch})
        
        avg_loss = epoch_loss / len(dataloader)
        # print epoch result nicely with tqdm.write so it doesn't mess up the progress bar.
        tqdm.write(f"Epoch [{epoch+1}/{config.epochs}] Average Loss: {avg_loss:.6f}")
        wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch})
        
        # Optionally save a checkpoint every 50 epochs and log it.
        if (epoch + 1) % 50 == 0:
            ckpt_path = f"dit_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
    
    # Save final model.
    torch.save(model.state_dict(), "dit_final.pt")
    wandb.save("dit_final.pt")

if __name__ == "__main__":
    train()
