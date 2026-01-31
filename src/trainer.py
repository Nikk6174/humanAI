import lightning as L
import torch
import torch.optim as optim
from torchmetrics.text import CharErrorRate, WordErrorRate
from src.architecture import RenAIssanceCRNN
from src.loss import FocalCTCLoss

class OCRTask(L.LightningModule):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.vocab = vocab
        self.idx2char = {idx + 1: char for idx, char in enumerate(vocab)}
        
        self.model = RenAIssanceCRNN(
            vocab_size=len(vocab),
            backbone_name=cfg.model.backbone,
            hidden_size=cfg.model.hidden_size,
            dropout=cfg.model.dropout
        )
        
        self.criterion = FocalCTCLoss(gamma=cfg.train.focal_gamma)
        self.cer = CharErrorRate()
        self.wer = WordErrorRate()

    def forward(self, x):
        return self.model(x)

    def _decode(self, log_probs, input_lengths):
        # Greedy CTC Decoder: Merges repeated characters and ignores the blank token (0)
        preds = torch.argmax(log_probs, dim=2).transpose(0, 1) 
        decoded_strs = []
        
        for i, pred in enumerate(preds):
            pred = pred[:input_lengths[i]]
            pred_str = ""
            last_char = 0
            for p in pred:
                p = p.item()
                if p != last_char and p != 0:
                    pred_str += self.idx2char[p]
                last_char = p
            decoded_strs.append(pred_str)
        return decoded_strs

    def training_step(self, batch, batch_idx):
        log_probs = self(batch['image'])
        loss = self.criterion(log_probs, batch['label'], batch['input_length'], batch['target_length'])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        log_probs = self(batch['image'])
        loss = self.criterion(log_probs, batch['label'], batch['input_length'], batch['target_length'])
        
        preds = self._decode(log_probs, batch['input_length'])
        self.cer.update(preds, batch['text'])
        self.wer.update(preds, batch['text'])
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_cer', self.cer, prog_bar=True)
        
        if batch_idx == 0:
            print(f"\nExample Prediction: '{preds[0]}' | Truth: '{batch['text'][0]}'")

    def configure_optimizers(self):
        # OneCycleLR provides a warm-up and cosine annealing for faster convergence
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.train.lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.train.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }