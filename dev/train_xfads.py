import os
import torch
import pytorch_lightning as lightning

from hydra import compose, initialize
from sklearn.linear_model import Ridge
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from xfads.prob_utils import estimate_poisson_rate_bias
from xfads.smoothers.lightning_trainers import LightningMonkeyReaching
from xfads.ssm_modules.prebuilt_models import create_xfads_poisson_log_link, create_xfads_poisson_log_link_w_input

import torch, gc

gc.collect()
torch.cuda.empty_cache()


def run_experiment(K: int, seed: int, data_splits_path: str, base_log_dir: str, base_ckpt_dir: str, cfg):
    """Train one experiment with given latent dimension K and seed."""

    # --- override parameters in cfg ---
    cfg.seed = seed
    cfg.n_latents = K
    cfg.n_latents_read = K
    cfg.rank_local = min(K, 15)
    cfg.rank_backward = 3 if K <= 15 else cfg.rank_backward
    lightning.seed_everything(cfg.seed, workers=True)

    print(cfg)

    # --- load data ---
    train_data = torch.load(data_splits_path + f"/data_train_{cfg.bin_sz_ms}ms_cor.pt")
    valid_data = torch.load(data_splits_path + f"/data_valid_{cfg.bin_sz_ms}ms_cor.pt")
    test_data = torch.load(data_splits_path + f"/data_test_{cfg.bin_sz_ms}ms_cor.pt")

    y_train_obs = train_data["y_obs"].float().to(cfg.data_device)
    y_valid_obs = valid_data["y_obs"].float().to(cfg.data_device)
    y_test_obs = test_data["y_obs"].float().to(cfg.data_device)

    train_input = train_data["input"].float().to(cfg.data_device)
    valid_input = valid_data["input"].float().to(cfg.data_device)
    test_input = test_data["input"].float().to(cfg.data_device)

    vel_train = train_data["velocity"].float().to(cfg.data_device)
    vel_valid = valid_data["velocity"].float().to(cfg.data_device)
    vel_test = test_data["velocity"].float().to(cfg.data_device)

    _, n_bins, n_neurons_obs = y_train_obs.shape

    # --- dataloaders ---
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_train_obs, vel_train, train_input),
        batch_size=cfg.batch_sz, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_valid_obs, vel_valid, valid_input),
        batch_size=y_valid_obs.shape[0], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_test_obs, vel_test, test_input),
        batch_size=y_test_obs.shape[0], shuffle=False
    )

    # --- model ---
    ssm = create_xfads_poisson_log_link_w_input(cfg, n_neurons_obs, 3, train_loader, model_type="c")
    seq_vae = LightningMonkeyReaching(ssm, cfg, n_bins, bin_prd_start=15, use_input=True)

    # --- logging and checkpoints ---
    log_dir = os.path.join(base_log_dir, f"{K}d/seed_{seed}")
    ckpt_dir = os.path.join(base_ckpt_dir, f"{K}d/seed_{seed}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_logger = lightning.loggers.CSVLogger(log_dir, name="training_logs", version='smoother_causal')

    ckpt_callback = lightning.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="r2_valid_enc",
        mode="max",
        dirpath=ckpt_dir,
        save_last=True,
        filename="{epoch:03d}_{valid_loss:.2f}_{r2_valid_enc:.2f}",
    )

    # --- trainer ---
    trainer = lightning.Trainer(
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.0,
        default_root_dir="lightning/",
        callbacks=[ckpt_callback],
        logger=csv_logger,
        devices=2,
        accelerator="cuda",
    )

    # --- train & test ---
    trainer.fit(model=seq_vae, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")

    # free GPU memory between runs
    import gc
    del trainer, seq_vae, ssm
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":

    import argparse
    from hydra import compose, initialize

    # --- parse command-line arguments ---
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--n_dims", type=int, nargs="+", required=True, help="List of latent dimensions, e.g. --n_dims 5 10 15 20")
    #args = parser.parse_args()

    #dims = args.n_dims
    dims = [40]

    # --- initialize config ---
    try:
        initialize(version_base=None, config_path="", job_name="monkey_reaching")
    except ValueError:
        pass  # already initialized
    cfg = compose(config_name="config")
    print(cfg)

    # --- use single fixed seed from cfg ---
    seed = cfg.seed

    animal = 1
    session = 3

    data_splits_path = f"./aligned_data/animal_{animal}/session_{session}"
    base_log_dir = f"/home/makki/forked/monkey_reaching_data_analysis/dev/logs/smoother/causal/animal_{animal}/session_{session}"
    base_ckpt_dir = f"/home/makki/forked/monkey_reaching_data_analysis/dev/ckpts/smoother/causal/animal_{animal}/session_{session}"

    os.makedirs(base_log_dir, exist_ok=True)
    os.makedirs(base_ckpt_dir, exist_ok=True)

    # --- sequential sweep ---
    for K in dims:
        print(f"\n=== Running experiment K={K}, seed={seed}, animal: {animal}, session: {session} ===\n")
        run_experiment(K, seed, data_splits_path, base_log_dir, base_ckpt_dir, cfg)
