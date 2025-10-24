import torch
torch.set_float32_matmul_precision('medium')
from torch import nn
from torch.utils.data import DataLoader, Dataset
from lightning import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch.nn.functional as F
import random
import zstandard
import os
import glob
import io
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
import rioxarray
from torch.utils.data import TensorDataset, DataLoader
import json
from preprocessing import load_and_normalise, decompress_zst_pt, get_file_list, collate_fn

class DownscalingDataset(Dataset):
    def __init__(self, file_list, static_vars, low_2mt_mean, low_2mt_std):
        self.file_list = file_list  # List of (high_file, low_file, date)
        self.static_vars = static_vars  # for a list of static variables, refer the preprocessing file which loads and normalises these datasets
        self.low_2mt_mean = low_2mt_mean
        self.low_2mt_std = low_2mt_std
        self.samples = []
        # Each file contains 24 hourly samples
        for hf, lf, date in self.file_list:
            for hour in range(24):
                self.samples.append((hf, lf, hour))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hf, lf, hour = self.samples[idx]
        # Load high-res and low-res data for the given hour
        high_data = decompress_zst_pt(hf)
        low_data = decompress_zst_pt(lf)
        # Get t2m fields
        high_t2m = high_data[hour]["2mT"].float()  # [672, 576]
        low_t2m = low_data[hour]["2mT"].float()    # [84, 72]. : Super resolution factor == 8
        # Standardise low-res t2m

        high_t2m = high_t2m.unsqueeze(0)  # [1, 672, 576]
        low_t2m = low_t2m.unsqueeze(0)    # [1, 84, 72]
        # Static vars already normalised and shaped: [1, 672, 576] or [bands, 672, 576]
        dem = self.static_vars["dem"].unsqueeze(0)  # [1, 672, 576]
        lat = self.static_vars["lat"].unsqueeze(0)  # [1, 672, 576]
        lc = self.static_vars["lc"]                 # [bands, 672, 576]
        return {
            "low_2mT": low_t2m,        # [1, 84, 72]
            "high_2mT": high_t2m,      # [1, 672, 576]
            "dem": dem,                # [1, 672, 576]
            "lat": lat,                # [1, 672, 576]
            "lc": lc                   # [bands, 672, 576]
        }
    
class DownscalingDataModule(LightningDataModule):
    def __init__(self, batch_size, val_frac, test_frac, num_workers, static_dir, save_stats_json):
        super().__init__()
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.num_workers = num_workers
        self.static_dir = static_dir
        self.save_stats_json = save_stats_json

    def setup(self, stage=None):
        static_vars, stats = load_and_normalise(
            self.static_dir,
            val_frac=self.val_frac,
            test_frac=self.test_frac,
            save_stats_json=self.save_stats_json
        )
        file_list = get_file_list()
        dataset = DownscalingDataset(
            file_list, static_vars,
            low_2mt_mean=stats["low_2mt_mean"],
            low_2mt_std=stats["low_2mt_std"]
        )
        N = len(dataset)
        n_val = int(self.val_frac * N)
        n_test = int(self.test_frac * N)
        n_train = N - n_val - n_test
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

    @property
    def test_dataset(self):
        return self.test_set.dataset


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size, padding=1)
        self.final = nn.Conv2d(channels[1], out_channels, 1)
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        output = self.final(h)
        return output


class UNetLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module = None,
        lr: float = 1e-3,
        optimizer: dict = None,
        scheduler: dict = None,
        loss_fn: torch.nn.Module = None,
        ckpt_path: str = None,
        ignore_keys: list = []
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'loss_fn'])
        self.net = net if net is not None else UNet(in_channels=19, out_channels=1)
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.lr = lr
        self.hparams.optimizer = optimizer
        self.hparams.scheduler = scheduler

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch):
        fuzzy_input, sharp_target = batch
        pred = self.forward(fuzzy_input)
        loss = self.loss_fn(pred, sharp_target)
        return loss, pred

    def training_step(self, batch, batch_idx):
        loss, pred = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_cfg = self.hparams.get("optimizer", None)
        sch_cfg = self.hparams.get("scheduler", None)

        if opt_cfg and opt_cfg["type"] == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=opt_cfg.get("lr", self.lr),
                betas=tuple(opt_cfg.get("betas", (0.5, 0.9))),
                weight_decay=opt_cfg.get("weight_decay", 1e-3)
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=opt_cfg.get("lr", self.lr) if opt_cfg else self.lr,
                weight_decay=opt_cfg.get("weight_decay", 1e-4) if opt_cfg else 1e-4
            )

        if sch_cfg and sch_cfg["type"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=sch_cfg.get("patience", 2),
                factor=sch_cfg.get("factor", 0.25)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": sch_cfg.get("monitor", "val/loss"),
                    "frequency": 1,
                },
            }
        else:
            return optimizer
    


    
def kl_from_standard_normal(mean, log_var):
    """Calculate KL divergence from standard normal."""
    kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
    return kl.mean()

def sample_from_standard_normal(mean, log_var, num=None):
    """Sample from normal distribution using reparameterisation trick."""
    std = log_var.mul(0.5).exp()
    shape = mean.shape
    if num is not None:
        shape = shape[:1] + (num,) + shape[1:]
        mean = mean[:, None, ...]
        std = std[:, None, ...]
    return mean + std * torch.randn(shape, device=mean.device)




class VAE(nn.Module):
    def __init__(
        self, 
        latent_dim=32, 
        lr=1e-3,
        kl_weight=0.001,
        input_channels=1,
        input_height=672,
        input_width=576,
        encoder_channels=[8],
        decoder_channels=[8],
        optimizer=None,
        scheduler=None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = VAE(
            latent_dim, input_channels, input_height, input_width, encoder_channels, decoder_channels
        )
        self.lr = lr
        self.kl_weight = kl_weight
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels[0], 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.encoded_size = encoder_channels[0] * (input_height // 2) * (input_width // 2)
        self.fc_mu = nn.Linear(self.encoded_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.encoded_size)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (decoder_channels[0], input_height // 2, input_width // 2)),
            nn.ConvTranspose2d(decoder_channels[0], input_channels, 2, stride=2),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Uses the corrected sample_from_standard_normal
        return sample_from_standard_normal(mu, logvar)

    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x, sample_posterior=True):
        mu, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        recon = self.decode(z)
        return recon, mu, logvar
    




class VAELitModule(LightningModule):
    def __init__(
        self, 
        latent_dim=32, 
        lr=1e-3,
        kl_weight=0.001,
        input_channels=1,
        input_height=672,
        input_width=576,
        encoder_channels=[8],
        decoder_channels=[8],
        optimizer=None,
        scheduler=None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = VAE(latent_dim, input_channels, input_height, input_width, encoder_channels, decoder_channels)
        self.lr = lr
        self.kl_weight = kl_weight
        self.hparams.optimizer = optimizer
        self.hparams.scheduler = scheduler
        
    def forward(self, x, sample_posterior=True):
        return self.model(x, sample_posterior)
    
    def _loss(self, batch):
        noisy, clean = batch
        recon, mu, logvar = self.forward(clean)
        
        # Recon loss
        recon_loss = F.l1_loss(recon, clean)
        
        # KL divergence loss
        kl_loss = kl_from_standard_normal(mu, logvar)
        
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
        
    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, kl_loss = self._loss(batch)
        
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, kl_loss = self._loss(batch)
        
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/recon_loss", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/kl_loss", kl_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        total_loss, recon_loss, kl_loss = self._loss(batch)
        
        self.log("test/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/recon_loss", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/kl_loss", kl_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss
        
    def configure_optimizers(self):
        # Example: using self.hparams for optimizer/scheduler settings from config
        opt_cfg = self.hparams.get("optimizer", None)
        sch_cfg = self.hparams.get("scheduler", None)

        if opt_cfg and opt_cfg["type"] == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=opt_cfg.get("lr", self.lr),
                betas=tuple(opt_cfg.get("betas", (0.5, 0.9))),
                weight_decay=opt_cfg.get("weight_decay", 1e-3)
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=opt_cfg.get("lr", self.lr) if opt_cfg else self.lr,
                weight_decay=opt_cfg.get("weight_decay", 1e-4) if opt_cfg else 1e-4
            )

        if sch_cfg and sch_cfg["type"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=sch_cfg.get("patience", 3),
                factor=sch_cfg.get("factor", 0.25)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": sch_cfg.get("monitor", "val/recon_loss"),
                    "frequency": 1,
                },
            }
        else:
            return optimizer
        


        

class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, num_layers=4):  # always make sure to have a config file ,, these are just default in case no information is provided in config.
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        layers = []
        layers.append(nn.Linear(latent_dim * 2, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_noisy, timestep=None):
        if timestep is None:
            timestep = torch.zeros(z_noisy.shape[0], 1, device=z_noisy.device)
        if timestep.max() > 1.0:
            timestep = timestep / 1000.0
        t_embed = self.time_embed(timestep.float())
        x = torch.cat([z_noisy, t_embed], dim=-1)
        return self.net(x)
    


    
class LDMLitModule(LightningModule):
    def __init__(
        self, 
        vae, 
        latent_dim=64, 
        lr=1e-4,
        num_timesteps=50,
        noise_schedule="linear",
        loss_type="l2",
        hidden_dim=128,
        num_layers=4,
        optimizer=None,
        scheduler=None,
        **kwargs
    ):
        super().__init__()
        # Save all hyperparameters except vae
        self.save_hyperparameters(logger=False, ignore=['vae'])
        self.vae = vae.model.requires_grad_(False)
        self.vae.eval()
        # Pass config values to LatentDenoiser
        self.denoiser = LatentDenoiser(latent_dim, hidden_dim, num_layers)
        self.lr = lr
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.register_noise_schedule(noise_schedule)
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def register_noise_schedule(self, schedule="linear"):
        if schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, self.num_timesteps)
        elif schedule == "cosine":
            timesteps = torch.arange(self.num_timesteps + 1) / self.num_timesteps
            alphas = torch.cos(timesteps * torch.pi / 2) ** 2
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise)

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.denoiser(x_noisy, t.unsqueeze(1))
        loss = self.loss_fn(predicted_noise, noise)
        return loss

    def forward(self, x):
        batch_size = x.shape[0]
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            z = self.vae.reparameterize(mu, logvar)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        loss = self.p_losses(z, t)
        return loss

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        loss = self.forward(clean)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        loss = self.forward(clean)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        noisy, clean = batch
        loss = self.forward(clean)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def sample(self, shape, num_steps=50):
        z = torch.randn(shape, device=self.device)
        timesteps = torch.linspace(self.num_timesteps-1, 0, num_steps, dtype=torch.long, device=self.device)
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(shape[0])
            predicted_noise = self.denoiser(z, t_batch.unsqueeze(1))
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
            pred_x0 = (z - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            if t > 0:
                noise = torch.randn_like(z) if i < len(timesteps) - 1 else 0
                z = (torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise)
            else:
                z = pred_x0
        return z

    @torch.no_grad()
    def generate_samples(self, num_samples=1):
        latent_shape = (num_samples, self.hparams.latent_dim)
        z_samples = self.sample(latent_shape)
        samples = self.vae.decode(z_samples)
        return samples

    def configure_optimizers(self):
        opt_cfg = self.hparams.get("optimizer", None)
        sch_cfg = self.hparams.get("scheduler", None)

        if opt_cfg and opt_cfg["type"] == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=opt_cfg.get("lr", self.lr),
                betas=tuple(opt_cfg.get("betas", (0.5, 0.9))),
                weight_decay=opt_cfg.get("weight_decay", 1e-3)
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=opt_cfg.get("lr", self.lr) if opt_cfg else self.lr,
                weight_decay=opt_cfg.get("weight_decay", 1e-4) if opt_cfg else 1e-4
            )

        if sch_cfg and sch_cfg["type"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=sch_cfg.get("patience", 5),
                factor=sch_cfg.get("factor", 0.5)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": sch_cfg.get("monitor", "val/loss"),
                    "frequency": 1,
                },
            }
        else:
            return optimizer



#For residuals, a res datamodule 

class ResidualsDataModule(LightningDataModule):
    def __init__(self, residuals, batch_size):
        super().__init__()
        self.residuals = residuals
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = TensorDataset(self.residuals, self.residuals)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    

    

def train_hierarchy(cfg):
    data_module = DownscalingDataModule(
        batch_size=cfg.data.batch_size,
        val_frac=cfg.data.val_split,
        test_frac=cfg.data.test_split,
        num_workers=cfg.data.num_workers,
        static_dir=cfg.paths.static_dir,  
        save_stats_json=os.path.join(cfg.paths.output_dir, "stats.json") 
    )
    data_module.setup()

    checkpoint_dir = cfg.paths.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # UNet
    unet_module = UNetLitModule(
    net=UNet(
        in_channels=cfg.model.unet.in_channels,
        out_channels=cfg.model.unet.out_channels,
        channels=cfg.model.unet.channels,
        kernel_size=cfg.model.unet.kernel_size,
    ),
    lr=cfg.model.unet.lr,
    optimizer=cfg.optimizer.unet,
    scheduler=cfg.scheduler.unet,
)
    unet_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir + "/unet",
        filename="unet-{epoch:02d}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )
    trainer_unet = Trainer(
        max_epochs=cfg.training.unet_epochs,
        accelerator=cfg.trainer.accelerator,
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        enable_checkpointing=True,
        logger=False,
        callbacks=[unet_checkpoint],
    )
    trainer_unet.fit(unet_module, datamodule=data_module)

    # Residuals
    residuals = []
    for batch in data_module.train_dataloader():
        fuzzy_input, sharp_target = batch
        with torch.no_grad():
            pred = unet_module.net(fuzzy_input)
            residual = sharp_target - pred
            residuals.append(residual)
    residuals = torch.cat(residuals, dim=0)
    print(f"Residuals shape: {residuals.shape}") #This is a debug step, as the VAE training had nans
    print("residual nans:",torch.isnan(residuals).any())

    # VAE
    vae_module = VAELitModule(
        latent_dim=cfg.model.vae.latent_dim,
        lr=cfg.model.vae.lr,
        kl_weight=cfg.model.vae.kl_weight,
        input_channels=cfg.model.vae.input_channels,
        input_height=cfg.model.vae.input_height,
        input_width=cfg.model.vae.input_width,
        encoder_channels=cfg.model.vae.encoder_channels,
        decoder_channels=cfg.model.vae.decoder_channels,
        optimizer=cfg.optimizer.vae,
        scheduler=cfg.scheduler.vae,
    )

    vae_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir + "/vae",
        filename="vae-{epoch:02d}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )
    residual_data_module = ResidualsDataModule(residuals, cfg.data.batch_size)
    residual_data_module.setup()

    
    trainer_vae = Trainer(
        max_epochs=cfg.training.vae_epochs,
        accelerator=cfg.trainer.accelerator,
        enable_checkpointing=True,
        logger=False,
        default_root_dir=checkpoint_dir + "/vae",
        callbacks=[vae_checkpoint],
    )
    trainer_vae.fit(vae_module, datamodule=residual_data_module)

    # LDM
    ldm_module = LDMLitModule(
        vae=vae_module,
        latent_dim=cfg.model.ldm.latent_dim,
        lr=cfg.model.ldm.lr,
        num_timesteps=cfg.model.ldm.num_timesteps,
        noise_schedule=cfg.model.ldm.noise_schedule,
        loss_type=cfg.model.ldm.loss_type,
        hidden_dim=cfg.model.ldm.hidden_dim,
        num_layers=cfg.model.ldm.num_layers,
        optimizer=cfg.optimizer.ldm,
        scheduler=cfg.scheduler.ldm,
    )
    ldm_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir + "/ldm",
        filename="ldm-{epoch:02d}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )
    trainer_ldm = Trainer(
        max_epochs=cfg.training.ldm_epochs,
        accelerator=cfg.trainer.accelerator,
        enable_checkpointing=True,
        logger=False,
        default_root_dir=checkpoint_dir + "/ldm",
        callbacks=[ldm_checkpoint],
    )
    trainer_ldm.fit(ldm_module, datamodule=residual_data_module)



if __name__ == "__main__":
    cfg = OmegaConf.load("conf/config_experiments.yaml")
    train_hierarchy(cfg)