import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalCTCLoss(nn.Module):
    def __init__(self, blank=0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Standard CTC loss represents -log(p)
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        
        # p is the probability of the correct sequence
        p = torch.exp(-ctc_loss)
        
        # Focal weighting: (1 - p)^gamma down-weights "easy" examples (high p)
        # This forces the model to prioritize rare characters and difficult handwriting.
        focal_weight = (1 - p) ** self.gamma
        loss = focal_weight * ctc_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss