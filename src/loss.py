import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalCTCLoss(nn.Module):
    """
    Focal CTC Loss: A modification of CTC Loss that focuses training on hard examples.
    Formula: Loss = (1 - p)^gamma * log(p)
    Where 'p' is the probability of the correct class.
    
    This helps the model learn rare characters (like renaissance 'ñ') that usually
    get drowned out by common letters like 'e' or 'a'.
    """
    def __init__(self, blank=0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # 1. Calculate Standard CTC Loss
        # log_probs: [Time, Batch, Class]
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        
        # 2. Calculate Probability of the prediction (p = exp(-loss))
        # Since CTC loss is essentially -log(p(target)), then p = exp(-ctc_loss)
        p = torch.exp(-ctc_loss)
        
        # 3. Apply Focal Weighting: (1 - p)^gamma
        # If model is 99% sure (p=0.99), weight is (0.01)^2 ~= 0.0001 (Ignore it)
        # If model is 10% sure (p=0.10), weight is (0.9)^2 ~= 0.81 (Focus on it!)
        focal_weight = (1 - p) ** self.gamma
        
        # 4. Final Loss
        loss = focal_weight * ctc_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss