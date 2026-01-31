# RenAIssance-OCR: Automating Historical Text Recognition

> **Note**  
> This research initiative is currently in the **active development phase**. The architecture and methodologies described below represent ongoing efforts to bridge the gap between computer vision and historical linguistics.

---

## 📜 Abstract

**Automating text recognition and transcription of historical documents with weighted convolutional–recurrent architectures and LLM models.**

The transliteration of text from centuries-old works represents a research area that is largely underserved by current commercial tools. While resources such as Adobe Acrobat’s OCR can effectively process clearly printed modern sources, they are frequently incapable of extracting textual data from early forms of print—let alone handwritten manuscripts.

**RenAIssance-OCR** focuses on the application of hybrid end-to-end models based on **weighted convolutional–recurrent architectures (CNN–RNN)** and **Large Language Models (LLMs)** to accurately recognize text in **seventeenth-century Spanish printed sources**.

---

## 🏗️ Model Architecture

This project implements a custom **CRNN (Convolutional Recurrent Neural Network)** designed specifically to handle the noise, degradation, and typographic irregularities of historical fonts. The architecture, defined as `RenAIssanceCRNN`, follows a modular design that enables interchangeable backbones and scalable complexity.

### 1. Feature Extraction (Vision Backbone)

A CNN backbone is used to extract spatial features from input images. The system is intentionally model-agnostic and leverages **timm (PyTorch Image Models)** to dynamically load different architectures.

- **Current Baseline:** ResNet-18  
- **Advanced Target:** ConvNeXt-Large (MLP-based), optimized for capturing high-fidelity details in degraded ink and uneven print impressions

---

### 2. The “Neck”: Feature Aggregation

A critical component of the architecture is the **Neck layer**. This `Conv2d` layer compresses the vertical dimension of the feature map, transforming 2D visual representations into a **1D sequential format** suitable for recurrent processing.

This stage acts as a **weighted learning mechanism**, emphasizing salient character-level features prior to sequence modeling.

---

### 3. Sequence Modeling (BiLSTM)

Sequential features are processed using a **Bidirectional LSTM (Long Short-Term Memory)** network.

- **Bidirectionality:** Enables the model to leverage both past and future context, improving disambiguation between visually similar characters (e.g., faint *e* vs *c*).
- **Depth:** A 2-layer stacked LSTM captures higher-order linguistic dependencies common in historical text.

---

### 4. Connectionist Temporal Classification (CTC)

The output head projects LSTM features to the vocabulary space. **CTC Loss (Connectionist Temporal Classification)** is used to align predicted sequences with ground-truth transcriptions, removing the need for pre-segmented character bounding boxes.

---

## 🛠️ Tech Stack & Implementation Details

The project is built on a **research-grade, reproducible stack** designed for scalability and experimentation:

- **Framework:** PyTorch & PyTorch Lightning (multi-GPU distributed training)
- **Configuration:** Hydra & OmegaConf for managing complex experimental setups (e.g., backbone swaps, hardware profiles)
- **Experiment Tracking:** Weights & Biases (WandB) for monitoring training dynamics, loss curves, and Character Error Rate (CER)
- **Data Augmentation:** Albumentations to simulate historical degradation (noise, blur, ink bleed, contrast shifts)
- **Evaluation Metrics:** torchmetrics and editdistance for precise transcription accuracy measurement

---

## 🚀 Research Roadmap & Objectives

The long-term objective is to move beyond standard OCR limitations through the following research directions:

1. **Hybrid End-to-End Modeling**  
   Finalize a robust CNN–RNN pipeline capable of accurate recognition on non-standard, early-modern printed text.

2. **Weighted Learning Techniques**  
   Improve recognition of rare letterforms, diacritics, ligatures, and symbols specific to Renaissance-era Spanish.

3. **Lexicon-Constrained Decoding**  
   Introduce constrained beam search decoding grounded in a curated Renaissance Spanish lexicon to reduce hallucinated outputs and improve word-level accuracy.

4. **LLM-Based Refinement**  
   Integrate LLMs (e.g., Gemini 1.5) as a post-processing layer to correct semantic inconsistencies and enhance final transcription quality.

---

## 📂 Project Structure

```plaintext
├── configs/               # Hydra configuration files
│   ├── model/             # Backbone configurations (ConvNeXt, ResNet)
│   └── hardware/          # Compute profiles (Laptop vs Cluster)
├── src/
│   ├── architecture.py    # RenAIssanceCRNN & visual modules
│   ├── dataset.py         # Custom OCRDataset with resizing & padding logic
│   └── trainer.py         # PyTorch LightningModule implementation
└── train.py               # Training entry point
