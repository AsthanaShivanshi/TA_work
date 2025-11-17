from multiprocessing import context
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from lightning import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from preprocessing import load_and_normalise, decompress_zst_pt, get_file_list, collate_fn


class DownscalingDataset(Dataset):
    def __init__(self, file_list, static_vars, low_2mt_mean, low_2mt_std):
        self.file_list = file_list
        self.static_vars = static_vars
        self.low_2mt_mean = low_2mt_mean
        self.low_2mt_std = low_2mt_std
        self.samples = []
        for hf, lf, date in self.file_list:
            for hour in range(24):
                self.samples.append((hf, lf, hour))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hf, lf, hour = self.samples[idx]
        high_data = decompress_zst_pt(hf)
        low_data = decompress_zst_pt(lf)
        high_t2m = high_data[hour]["2mT"].float().unsqueeze(0)
        low_t2m = low_data[hour]["2mT"].float().unsqueeze(0)
        dem = self.static_vars["dem"].unsqueeze(0)
        lat = self.static_vars["lat"].unsqueeze(0)
        lc = self.static_vars["lc"]
        fuzzy_input = torch.cat([
            F.interpolate(low_t2m.unsqueeze(0), size=high_t2m.shape[-2:], mode='bilinear', align_corners=False).squeeze(0),
            dem, lat, lc
        ], dim=0)
        return fuzzy_input, high_t2m



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
        self.train_set = torch.utils.data.Subset(dataset, range(0, n_train))
        self.val_set = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
        self.test_set = torch.utils.data.Subset(dataset, range(n_train + n_val, N))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

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
        in_channels: int = 19,
        out_channels: int = 1,
        channels: list = [32, 16],
        kernel_size: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'loss_fn'])
        self.net = net if net is not None else UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            kernel_size=kernel_size
        )
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


def kl_divergence(mean, log_var):
    kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
    return kl.mean()

class VAE(nn.Module):
    def __init__(self, latent_dim=36, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (hidden_dim, 1, 1)),
            nn.ConvTranspose2d(hidden_dim, 32, 8, stride=8),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 84, stride=84),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_decode(z)
        x = self.decoder(h)
        # x: [B, 1, H, W] (likely [672, 672])
        x = F.interpolate(x, size=(672, 576), mode='bilinear', align_corners=False)
        return x

    def forward(self, x, sample_posterior=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        recon = self.decode(z)
        return recon, mu, logvar

class VAELitModule(LightningModule):
    def __init__(self, vae, unet_module, kl_weight=0.001, lr=1e-3):
        super().__init__()
        self.vae = vae
        self.unet = unet_module
        self.kl_weight = kl_weight
        self.lr = lr

    def training_step(self, batch, batch_idx):
        fuzzy_input, sharp_target = batch
        with torch.no_grad():
            unet_pred = self.unet(fuzzy_input)
        residual = sharp_target - unet_pred
        # Only use the UNet-predicted temperature channel as condition
        condition = unet_pred[:, 0:1]
        recon, mu, logvar = self.vae(condition)
        recon_loss = F.l1_loss(recon, residual)
        kl_loss = kl_divergence(mu, logvar)
        total_loss = recon_loss + self.kl_weight * kl_loss
        self.log("train/loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        fuzzy_input, sharp_target = batch
        with torch.no_grad():
            unet_pred = self.unet(fuzzy_input)
        residual = sharp_target - unet_pred
        condition = unet_pred[:, 0:1]
        recon, mu, logvar = self.vae(condition)
        recon_loss = F.l1_loss(recon, residual)
        kl_loss = kl_divergence(mu, logvar)
        total_loss = recon_loss + self.kl_weight * kl_loss
        self.log("val/loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LatentDenoiser(nn.Module):
    def __init__(self, in_channels, out_channels, channels, kernel_size, context_channels=0):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels[0]),
            nn.ReLU(),
            nn.Linear(channels[0], channels[0])
        )
        self.unet = UNet(
            in_channels=in_channels + 2 + context_channels,  # +context_channels
            out_channels=out_channels,
            channels=channels,
            kernel_size=kernel_size
        )

    def forward(self, z_noisy, unet_pred, timestep=None, context=None):
        if timestep is None:
            timestep = torch.zeros(z_noisy.shape[0], 1, device=z_noisy.device)
        B, _, H, W = z_noisy.shape
        t_embed = self.time_embed(timestep.float())
        t_map = t_embed[:, :1].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, H, W)
        inputs = [z_noisy, unet_pred, t_map]
        if context is not None:
            inputs.append(context)
        x = torch.cat(inputs, dim=1)
        return self.unet(x)

class LDMLitModule(LightningModule):
    def __init__(self, vae, latent_dim=36, conditioner=None, latent_height=6, latent_width=6, lr=1e-4, num_timesteps=50, noise_schedule="linear", loss_type="l2", hidden_dim=128, num_layers=4, param_type="eps", optimizer=None, scheduler=None):
        super().__init__()
        assert latent_dim == latent_height * latent_width,(f"latent_dim ({latent_dim}) must equal latent_height*latent_width ({latent_height}*{latent_width})")
        self.save_hyperparameters(logger=False, ignore=['vae'])
        self.vae = vae.vae.requires_grad_(False)
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.vae.eval()
        self.unet = vae.unet
        self.conditioner = conditioner
        self.denoiser = LatentDenoiser(1, 1, [hidden_dim]*num_layers, 3, context_channels=1)
        self.lr = lr
        self.num_timesteps = num_timesteps
        self.loss_fn = {"l1": nn.L1Loss(), "l2": nn.MSELoss()}[loss_type]
        self.param_type = param_type
        self.register_noise_schedule(noise_schedule)
        self.hparams.optimizer = optimizer
        self.hparams.scheduler = scheduler




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
        # Ensure t is [B] and expand to [B, 1, 1, 1] for broadcasting
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, x_start, unet_pred, t, noise=None, context=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        B = x_noisy.shape[0]
        x_noisy_4d = x_noisy.view(B, 1, self.latent_height, self.latent_width)
        unet_pred_ds = F.interpolate(unet_pred, size=(self.latent_height, self.latent_width), mode='bilinear', align_corners=False)
        context_ds = None
        if self.conditioner is not None and context is not None:
            context_ds = self.conditioner(context)

        # Only call once, with context
        predicted = self.denoiser(x_noisy_4d, unet_pred_ds, t.unsqueeze(1).to(x_noisy.device), context=context_ds)

        # param can be "eps", "x0", or "v"
        if self.param_type == "eps":
            target = noise.view(B, 1, self.latent_height, self.latent_width)
        elif self.param_type == "x0":
            target = x_start.view(B, 1, self.latent_height, self.latent_width)
        elif self.param_type == "v":
            alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            target = alpha_t * noise.view(B, 1, self.latent_height, self.latent_width) - sigma_t * x_start.view(B, 1, self.latent_height, self.latent_width)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")

        return self.loss_fn(predicted, target)

    def forward(self, x, unet_pred, context=None):
        with torch.no_grad():
            mu, logvar = self.vae.encode(unet_pred[:, 0:1])
            z = self.vae.reparameterize(mu, logvar)
        print("z shape before reshape:", z.shape)  # Debug print
        assert z.shape[1] * z.shape[0] != self.latent_height * self.latent_width, (
            f"Latent shape mismatch: got {z.shape}, expected ({-1}, {self.latent_height * self.latent_width})"
        )
        z = z.view(-1, 1, self.latent_height, self.latent_width)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device)
        return self.p_losses(z, z, t, context=context)
        

    def training_step(self, batch, batch_idx):
        fuzzy_input, sharp_target = batch
        with torch.no_grad():
            unet_pred = self.unet(fuzzy_input)
        # Only pass low-res temperature (assumed to be channel 0) as context
        context = fuzzy_input[:, 0:1]
        loss = self.forward(fuzzy_input, unet_pred, context=context)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        fuzzy_input, sharp_target = batch
        with torch.no_grad():
            unet_pred = self.unet(fuzzy_input)
        context = fuzzy_input[:, 0:1]
        loss = self.forward(fuzzy_input, unet_pred, context=context)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        fuzzy_input, sharp_target = batch
        with torch.no_grad():
            unet_pred = self.unet(fuzzy_input)
        context = fuzzy_input[:, 0:1]
        loss = self.forward(fuzzy_input, unet_pred, context=context)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt_cfg = self.hparams.get("optimizer", None)
        sch_cfg = self.hparams.get("scheduler", None)
        if opt_cfg and opt_cfg.get("type") == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=opt_cfg.get("lr", self.lr), betas=tuple(opt_cfg.get("betas", (0.5, 0.9))), weight_decay=opt_cfg.get("weight_decay", 1e-3))
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=opt_cfg.get("lr", self.lr) if opt_cfg else self.lr, weight_decay=opt_cfg.get("weight_decay", 1e-4) if opt_cfg else 1e-4)
        if sch_cfg and sch_cfg.get("type") == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=sch_cfg.get("patience", 5), factor=sch_cfg.get("factor", 0.5))
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": sch_cfg.get("monitor", "val/loss"), "frequency": 1}}
        return optimizer



#for conditional geeration, writing a simple CNN conditooner : 

class Conditioner(nn.Module):
    def __init__(self, in_channels, out_channels, latent_height, latent_width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.AdaptiveAvgPool2d((latent_height, latent_width))
        )

    def forward(self, x):
        return self.net(x)
    


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

    unet_ckpt_dir = os.path.join(checkpoint_dir, "unet")
    os.makedirs(unet_ckpt_dir, exist_ok=True)
    unet_ckpts = sorted([f for f in os.listdir(unet_ckpt_dir) if f.endswith(".ckpt")])
    if unet_ckpts:
        print("Loading pretrained UNet ckpt")
        unet_module = UNetLitModule.load_from_checkpoint(os.path.join(unet_ckpt_dir, unet_ckpts[-1]))
    else:
        print("Training UNet")
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
            dirpath=unet_ckpt_dir,
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

    vae_ckpt_dir = os.path.join(checkpoint_dir, "vae")
    os.makedirs(vae_ckpt_dir, exist_ok=True)
    vae_ckpts = sorted([f for f in os.listdir(vae_ckpt_dir) if f.endswith(".ckpt")])
    if vae_ckpts:
        print("Loading pretrained VAE ckpt")
        vae = VAE(latent_dim=cfg.model.vae.latent_dim,hidden_dim=cfg.model.vae.hidden_dim
)
        vae_module = VAELitModule.load_from_checkpoint(
    os.path.join(vae_ckpt_dir, vae_ckpts[-1]),
    vae=vae,
    unet_module=unet_module
)
    else:
        print("Training VAE")
        vae = VAE(
            latent_dim=cfg.model.vae.latent_dim,
            hidden_dim=cfg.model.vae.hidden_dim
        )
        vae_module = VAELitModule(
            vae=vae,
            unet_module=unet_module,
            kl_weight=cfg.model.vae.kl_weight,
            lr=cfg.model.vae.lr
        )
        vae_checkpoint = ModelCheckpoint(
            dirpath=vae_ckpt_dir,
            filename="vae-{epoch:02d}",
            save_top_k=1,
            monitor="val/loss",
            mode="min",
        )
        trainer_vae = Trainer(
            max_epochs=cfg.training.vae_epochs,
            accelerator=cfg.trainer.accelerator,
            enable_checkpointing=True,
            logger=False,
            default_root_dir=vae_ckpt_dir,
            callbacks=[vae_checkpoint],
        )
        trainer_vae.fit(vae_module, datamodule=data_module)

    ldm_ckpt_dir = os.path.join(checkpoint_dir, "ldm")
    os.makedirs(ldm_ckpt_dir, exist_ok=True)
    ldm_ckpts = sorted([f for f in os.listdir(ldm_ckpt_dir) if f.endswith(".ckpt")])
    if ldm_ckpts:
        print("Loading pretrained LDM ckpt")
        ldm_module = LDMLitModule.load_from_checkpoint(os.path.join(ldm_ckpt_dir, ldm_ckpts[-1]), vae=vae_module)
    else:
        print("Training LDM")


        conditioner= Conditioner(
            in_channels= 1,
            out_channels= 1,
            latent_height=cfg.model.ldm.latent_height,
            latent_width=cfg.model.ldm.latent_width
        ) 

        ldm_module = LDMLitModule(
            vae=vae_module,
            conditioner=conditioner,
            latent_dim=cfg.model.ldm.latent_dim,
            latent_height=cfg.model.ldm.latent_height,
            latent_width=cfg.model.ldm.latent_width,
            lr=cfg.model.ldm.lr,
            num_timesteps=cfg.model.ldm.num_timesteps,
            noise_schedule=cfg.model.ldm.noise_schedule,
            loss_type=cfg.model.ldm.loss_type,
            hidden_dim=cfg.model.ldm.hidden_dim,
            num_layers=cfg.model.ldm.num_layers,
            optimizer=cfg.optimizer.ldm,
            scheduler=cfg.scheduler.ldm,
            param_type=cfg.model.ldm.param_type,
        )
        ldm_checkpoint = ModelCheckpoint(
            dirpath=ldm_ckpt_dir,
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
            default_root_dir=ldm_ckpt_dir,
            callbacks=[ldm_checkpoint],
        )
        trainer_ldm.fit(ldm_module, datamodule=data_module)



if __name__ == "__main__":
    cfg = OmegaConf.load("conf/config_experiments.yaml")
    train_hierarchy(cfg)