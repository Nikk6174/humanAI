import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class VisualAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.shape
        # Generate spatial attention weights to emphasize character features
        scores = self.score(x) 
        attn_weights = self.softmax(scores.view(b, 1, -1)).view(b, 1, h, w)
        return x * attn_weights

class RenAIssanceCRNN(nn.Module):
    def __init__(self, vocab_size, backbone_name='convnext_large_mlp', pretrained=True, hidden_size=256, dropout=0.2):
        super().__init__()
        
        # Load backbone and dynamically determine output shapes
        self.feature_extractor = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=(-1,), in_chans=1
        )
        
        with torch.no_grad():
            dummy_out = self.feature_extractor(torch.randn(1, 1, 32, 1024))
            self.feature_channels = dummy_out[-1].shape[1]
            self.feature_height = dummy_out[-1].shape[2]

        # Neck: Squashes height dimension to convert 2D features into a 1D sequence
        self.neck = nn.Conv2d(self.feature_channels, hidden_size, kernel_size=(self.feature_height, 1))
        
        self.rnn = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=2,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        
        self.head = nn.Linear(hidden_size * 2, vocab_size + 1)

    def forward(self, x):
        features = self.feature_extractor(x)[-1]
        
        features = self.neck(features)
        features = features.squeeze(2).permute(0, 2, 1) # [Batch, Time, Hidden]
        
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(features)
        
        logits = self.head(rnn_out)
        
        # Transpose for PyTorch CTCLoss: [Time, Batch, Class]
        logits = logits.permute(1, 0, 2)
        return F.log_softmax(logits, dim=2)

if __name__ == "__main__":
    print("🧪 Running Sanity Check...")
    model = RenAIssanceCRNN(vocab_size=100, backbone_name='resnet18') 
    dummy_img = torch.randn(2, 1, 32, 512)
    output = model(dummy_img)
    print(f"🎉 Output Shape: {output.shape} (Time, Batch, Classes)")