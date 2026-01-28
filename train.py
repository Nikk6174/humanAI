import hydra
from omegaconf import DictConfig
import torch
import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from src.dataset import OCRDataset, custom_collate_fn
from src.trainer import OCRTask
import albumentations as A

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Set Seed for Reproducibility
    L.seed_everything(cfg.seed)
    
    # 2. Prepare Data
    # We use a simple charset for now. In Phase 2, we will load this from a file.
    VOCAB = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
    
    # Define Augmentations (Only for training)
    # Define Augmentations (Only for training)
    train_transform = A.Compose([
        # FIX: Removed 'alpha_affine=50' which causes the crash in new versions
        A.ElasticTransform(alpha=1, sigma=50, p=0.3), 
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
    ])

    print(f"📂 Loading Data from: {cfg.data_dir}")
    full_dataset = OCRDataset(
        csv_file=os.path.join(cfg.data_dir, "labels.csv"),
        root_dir=cfg.data_dir,
        vocab=VOCAB,
        transform=train_transform
    )

    # 90% Train / 10% Validation split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Disable augmentations for validation (important!)
    val_dataset.dataset.transform = None 

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.hardware.batch_size, 
        shuffle=True, 
        num_workers=cfg.hardware.num_workers,
        collate_fn=custom_collate_fn,
        persistent_workers=True if cfg.hardware.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.hardware.batch_size, 
        shuffle=False, 
        num_workers=cfg.hardware.num_workers,
        collate_fn=custom_collate_fn
    )

    # 3. Setup Model
    model = OCRTask(cfg, VOCAB)

    # 4. Setup Callbacks (The "Auto-Pilot" features)
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="renai-ocr-{epoch:02d}-{val_cer:.4f}",
        monitor="val_cer",
        mode="min",
        save_top_k=3
    )
    early_stop_cb = EarlyStopping(monitor="val_cer", patience=10, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 5. Setup Logger (WandB)
    # Set 'offline=True' if you don't have internet on the cluster
    wandb_logger = WandbLogger(project=cfg.project_name, log_model="all")

    # 6. Initialize Trainer
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        strategy=cfg.hardware.get("strategy", "auto"), # "ddp" for cluster, "auto" for laptop
        logger=wandb_logger,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        precision="16-mixed" if cfg.hardware.accelerator == "gpu" else 32, # Mixed Precision for Speed!
        gradient_clip_val=1.0 # Crucial for RNN stability
    )

    # 7. BLAST OFF 🚀
    print("🚀 Starting Training...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()