import lightning as L
import torch
import torch.optim as optim
from torchmetrics.text import CharErrorRate, WordErrorRate
from src.architecture import RenAIssanceCRNN
from src.loss import FocalCTCLoss

class OCRTask(L.LightningModule):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.save_hyperparameters() # Saves cfg to checkpoints automatically
        self.cfg = cfg
        self.vocab = vocab
        self.idx2char = {idx + 1: char for idx, char in enumerate(vocab)}
        
        # 1. Initialize Model (Dynamically swappable via Config)
        self.model = RenAIssanceCRNN(
            vocab_size=len(vocab),
            backbone_name=cfg.model.backbone,
            hidden_size=cfg.model.hidden_size,
            dropout=cfg.model.dropout
        )
        
        # 2. Loss Function (The Mathematical Flex)
        self.criterion = FocalCTCLoss(gamma=cfg.train.focal_gamma)
        
        # 3. Metrics (CER & WER)
        self.cer = CharErrorRate()
        self.wer = WordErrorRate()

    def forward(self, x):
        return self.model(x)

    def _decode(self, log_probs, input_lengths):
        """
        Greedy Decoder for logging predictions during training.
        (We will use Beam Search for the final evaluation, but Greedy is faster for training checks)
        """
        # log_probs shape: [Time, Batch, Class] -> argmax over Class
        preds = torch.argmax(log_probs, dim=2).transpose(0, 1) # [Batch, Time]
        decoded_strs = []
        
        for i, pred in enumerate(preds):
            pred = pred[:input_lengths[i]] # Ignore padding
            pred_str = ""
            last_char = 0
            for p in pred:
                p = p.item()
                # CTC Logic: Merge repeated characters and ignore blanks (0)
                if p != last_char and p != 0:
                    pred_str += self.idx2char[p]
                last_char = p
            decoded_strs.append(pred_str)
        return decoded_strs

    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        input_lengths = batch['input_length']
        target_lengths = batch['target_length']

        # Forward Pass
        log_probs = self(images) # [Time, Batch, Class]
        
        # Calculate Loss
        loss = self.criterion(log_probs, labels, input_lengths, target_lengths)
        
        # Log to WandB / TensorBoard
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label'] # Padded tensor [Batch, Max_Len]
        input_lengths = batch['input_length']
        target_lengths = batch['target_length']
        texts = batch['text'] # Original strings

        log_probs = self(images)
        loss = self.criterion(log_probs, labels, input_lengths, target_lengths)
        
        # Calculate Metrics
        preds = self._decode(log_probs, input_lengths)
        
        # Clean calculation of CER/WER
        self.cer.update(preds, texts)
        self.wer.update(preds, texts)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_cer', self.cer, prog_bar=True)
        
        # Log a few examples to see progress visually
        if batch_idx == 0:
            print(f"\nExample Prediction: '{preds[0]}' | Truth: '{texts[0]}'")

    def configure_optimizers(self):
        # The 'OneCycleLR' scheduler is the secret weapon for super-convergence
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.train.lr, weight_decay=1e-2)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.train.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1, # Warmup for first 10%
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }