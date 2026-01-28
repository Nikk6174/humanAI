import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class VisualAttention(nn.Module):
    """
    The 'Attention' Mechanism. 
    Allows the RNN to 'look back' at the image features to find specific curves
    when it's unsure about a character.
    """
    def __init__(self, channels, H, W):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.score = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2) # Attention over width

    def forward(self, x):
        # x: [Batch, Channel, Height, Width]
        b, c, h, w = x.shape
        
        # Calculate "Importance" map
        scores = self.score(x) 
        attn_weights = self.softmax(scores.view(b, 1, -1)).view(b, 1, h, w)
        
        # Apply attention to features
        out = x * attn_weights
        return out

class RenAIssanceCRNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 backbone_name='convnext_large_mlp', 
                 pretrained=True,
                 hidden_size=256,
                 dropout=0.2):
        super().__init__()
        
        # 1. THE EYES (Feature Extractor)
        print(f"🏗️ Initializing Backbone: {backbone_name}...")
        self.feature_extractor = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(-1,),
            in_chans=1  # <--- Ensure this is here!
        )
        
        # --- FIX IS HERE ---
        # Hack to dynamically find the output channels. 
        # MUST BE 1 CHANNEL (Grayscale) to match the model.
        dummy_input = torch.randn(1, 1, 32, 1024) 
        # -------------------

        with torch.no_grad():
            dummy_out = self.feature_extractor(dummy_input)
            self.feature_channels = dummy_out[-1].shape[1]
            self.feature_height = dummy_out[-1].shape[2]
            
        print(f"✅ Backbone Channels: {self.feature_channels}, Height: {self.feature_height}")

        # 2. THE NECK (Dimensionality Reduction)
        self.neck = nn.Conv2d(self.feature_channels, hidden_size, kernel_size=(self.feature_height, 1))
        
        # 3. THE BRAIN (Bidirectional LSTM)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # 4. THE MOUTH (Classifier)
        self.head = nn.Linear(hidden_size * 2, vocab_size + 1)

        
    def forward(self, x):
        # x shape: [Batch, 3, 32, Width]
        
        # 1. Extract Features
        # Output: [Batch, Channels, H', W']
        features = self.feature_extractor(x)[-1]
        
        # 2. Prepare for Sequence (Neck)
        # Squash Height: [Batch, Hidden, 1, W']
        features = self.neck(features)
        
        # 3. Reshape for RNN
        # Remove height dim: [Batch, Hidden, W']
        features = features.squeeze(2)
        # Permute to [Batch, W', Hidden] (Sequence First)
        features = features.permute(0, 2, 1)
        
        # 4. Sequence Modeling (RNN)
        # rnn_out: [Batch, W', Hidden*2]
        self.rnn.flatten_parameters() # Optimization for GPU
        rnn_out, _ = self.rnn(features)
        
        # 5. Prediction
        # logits: [Batch, W', Vocab+1]
        logits = self.head(rnn_out)
        
        # Re-arrange for PyTorch CTC Loss: [W', Batch, Vocab+1]
        # This is a specific requirement of torch.nn.CTCLoss
        logits = logits.permute(1, 0, 2)
        
        # Apply LogSoftmax for stability
        log_probs = F.log_softmax(logits, dim=2)
        
        return log_probs

if __name__ == "__main__":
    # --- SANITY CHECK (Run this on Laptop) ---
    print("🧪 Running Sanity Check on Laptop...")
    
    # Use a tiny backbone for testing
    model = RenAIssanceCRNN(vocab_size=100, backbone_name='resnet18') 
    
    # Create a dummy image (Batch 2, RGB, Height 32, Width 512)
    dummy_img = torch.randn(2, 3, 32, 512)
    
    # Forward pass
    output = model(dummy_img)
    
    print(f"\n🎉 Success!")
    print(f"Input Shape: {dummy_img.shape}")
    print(f"Output Shape: {output.shape} (Sequence_Length, Batch, Classes)")
    print("The code logic is valid. You can now swap 'resnet18' for 'convnext_large_mlp' on the cluster.")