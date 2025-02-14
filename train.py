"""
train.py

Train a diffusion model using DDPM on MNIST with the DiT model from dit.py.
This script:
  • Loads MNIST and converts the 28×28 grayscale images into 64×64 “RGB”
    images via a transform.
  • Sets up a DDPM forward process with a linear beta schedule.
  • At each iteration, randomly selects a timestep t, creates a noisy image
    using the closed‐form solution: 
         xₜ = √(ᾱₜ)·x₀ + √(1−ᾱₜ)·ε,
    and trains DiT to predict the added Gaussian noise ε.
  • Every so often, runs the reverse (sampling) process using the DDPM
    update step:
         xₜ₋₁ = (1/√αₜ)·[xₜ – (βₜ/√(1−ᾱₜ))·εθ(xₜ,t)] + √(βₜ)·z  (if t>0)
  • Logs losses and generated images to wandb under project "dit-ddpm-mnist".
  
Be sure that dit.py (which contains the DiT model and DiTConfig definition)
is in the same directory or otherwise on your PYTHONPATH.
"""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import wandb

# Import the DiT model and configuration from dit.py
from dit import DiT, DiTConfig

# ------------------------ Diffusion Schedule -------------------------


def get_cosine_schedule(T, device):
    """
    Create a cosine beta schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'
    """
    steps = torch.linspace(0, T-1, T, dtype=torch.float32, device=device)
    f_t = torch.cos(((steps / T) + 0.008) / (1.008) * math.pi * 0.5) ** 2
    alpha_bars = f_t / f_t[0]  # Ensure alpha_bar[0] = 1.0
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    betas = torch.cat([torch.tensor([1e-4], device=device), betas])

    # Compute regular alphas
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return betas, alphas, alpha_bars

# ----------------------- DDPM Sampling Function -----------------------


@torch.no_grad()
def sample_ddpm(model, T, betas, alphas, alpha_bars, device, sample_size, config):
    """
    Reverse diffusion: start from pure noise and iteratively denoise.
    Uses the DDPM update rule:
       μₜ = (1/√αₜ) * [xₜ - ((βₜ/√(1-ᾱₜ)) * εθ(xₜ,t)]
       Then: xₜ₋₁ = μₜ + √(βₜ)*z   for t > 0 (and no noise for t==0)

    Note that the model is conditioned on a tensor t (the timestep)
    which we set to a constant at each reverse step.
    """
    model.eval()  # use eval mode during sampling
    x = torch.randn(sample_size, config.in_channels,
                    config.input_size, config.input_size, device=device)

    for t in reversed(range(T)):
        t_tensor = torch.full(
            (sample_size,), t, device=device, dtype=torch.float32)
        # Get current scalar values from our schedules.
        beta = betas[t].item()
        alpha = alphas[t].item()
        alpha_bar = alpha_bars[t].item()

        # Predict noise using our trained model.
        eps_theta = model(x, t_tensor)

        # Compute DDPM mean update:
        # μₜ = (1/√αₜ) * (xₜ - ((βₜ/√(1-ᾱₜ)) * eps_theta))
        sqrt_alpha = math.sqrt(alpha)
        sqrt_one_minus_alpha_bar = math.sqrt(1 - alpha_bar)
        x = (1/sqrt_alpha) * (x - (beta / sqrt_one_minus_alpha_bar) * eps_theta)

        # Add noise if not the final step.
        if t > 0:
            noise = torch.randn_like(x)
            x = x + math.sqrt(beta) * noise

    model.train()
    return x

# --------------------- Main Training Loop ----------------------------


def main():
    # Use GPU if available.
    device = "cuda" if torch.cuda.is_available() else "mps"

    # Initialize wandb.
    wandb.init(project="dit-ddpm-mnist", config={
        "batch_size": 64,
        "learning_rate": 1e-4,
        "diffusion_steps": 200,
        "image_size": 64,
        "epochs": 100,
    })
    config_wandb = wandb.config

    # Transform for MNIST: Resize to 64x64, convert grayscale to 3-channel, scale to [-1,1].
    transform = transforms.Compose([
        transforms.Resize(config_wandb.image_size),
        # replicate channel to create an "RGB" image
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config_wandb.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

    # Instantiate DiT (our network) with the provided configuration.
    # Note: DiTConfig from dit.py uses default values for input_size=64 and in_channels=3.
    """
    {
          inputSize: 64, // 64x64 image
          patchSize: 8,
          inChannels: 3,
          dim: 64,
          depth: 6,
          dimHead: 32,
          mlpMult: 4,
          timeEmbDim: 64,
        };
        """
    model_config = DiTConfig(
        input_size=config_wandb.image_size,
        patch_size=8,
        in_channels=3,
        dim=256,
        depth=4,
        dim_head=64,
        mlp_mult=4,
        time_emb_dim=256
    )
    model = DiT(model_config).to(device)

    # Set up optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config_wandb.learning_rate)

    # Diffusion hyperparameters.
    # total timesteps for the forward process (e.g. 1000)
    T = config_wandb.diffusion_steps
    betas, alphas, alpha_bars = get_cosine_schedule(T, device)

    print("Starting training...")

    total_steps = 0
    sample_interval = 500  # generate samples every 500 training steps

    # Training loop.
    for epoch in range(config_wandb.epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)  # x: [B, 3, 64, 64] in [-1, 1]
            B = x.size(0)
            # Sample random diffusion timesteps for each image.
            t_int = torch.randint(0, T, (B,), device=device)
            t_float = t_int.float()

            # Sample standard Gaussian noise.
            noise = torch.randn_like(x)

            # Get corresponding ᾱ (alpha_bar) values and reshape for broadcasting.
            alpha_bar_t = alpha_bars[t_int].view(B, 1, 1, 1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

            # Create the noisy image: xₜ = √(ᾱₜ)*x₀ + √(1−ᾱₜ)*noise.
            x_noisy = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise

            # Predict the noise added (the training target) using the network.
            predicted_noise = model(x_noisy, t_float)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1

            # Log training loss every 100 steps.
            if total_steps % 100 == 0:
                print(
                    f"Epoch {epoch} Step {total_steps} Loss: {loss.item():.6f}")
                wandb.log({"loss": loss.item(), "step": total_steps})

            # Periodically, generate samples from the model.
            if total_steps % sample_interval == 0:
                sampled_images = sample_ddpm(
                    model, T, betas, alphas, alpha_bars, device, sample_size=16, config=model_config)
                # Denormalize images back to [0,1] for visualization.
                sampled_images = (sampled_images + 1) / 2.0
                grid = utils.make_grid(sampled_images, nrow=4)
                wandb.log({"generated_images": [wandb.Image(
                    grid, caption=f"Step {total_steps}")]})

        # Optionally save a checkpoint at the end of each epoch.
        torch.save(model.state_dict(), f"dit_ddpm_mnist_epoch{epoch}.pt")


if __name__ == "__main__":
    main()
