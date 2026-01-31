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
    L.seed_everything(cfg.seed)
    
    VOCAB = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
    
    train_transform = A.Compose([
        A.ElasticTransform(alpha=1, sigma=50, p=0.3), 
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
    ])

    # Load dataset and perform 90/10 train-validation split
    full_dataset = OCRDataset(
        csv_file=os.path.join(cfg.data_dir, "labels.csv"),
        root_dir=cfg.data_dir,
        vocab=VOCAB,
        transform=train_transform
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
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

    model = OCRTask(cfg, VOCAB)

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="renai-ocr-{epoch:02d}-{val_cer:.4f}",
        monitor="val_cer",
        mode="min",
        save_top_k=3
    )
    
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        strategy=cfg.hardware.get("strategy", "auto"),
        logger=WandbLogger(project=cfg.project_name, log_model="all"),
        callbacks=[checkpoint_cb, EarlyStopping(monitor="val_cer", patience=10), LearningRateMonitor(logging_interval='step')],
        # Use mixed precision for performance and gradient clipping for RNN stability
        precision="16-mixed" if cfg.hardware.accelerator == "gpu" else 32,
        gradient_clip_val=1.0 
    )

    print("🚀 Starting Training...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()