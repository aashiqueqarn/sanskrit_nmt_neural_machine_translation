# Sanskrit Neural Machine Translation (NMT)

A deep learning-based neural machine translation model that converts Sanskrit sentences to English using an LSTM-based sequence-to-sequence architecture with Luong attention mechanism.

## Project Overview

This project implements an encoder-decoder LSTM model trained on Sanskrit-English parallel corpus. The model uses SentencePiece tokenization and Luong attention for context generation.

**Key Features:**
- LSTM-based Seq2Seq architecture with attention mechanism
- Bidirectional LSTM encoder for better context understanding
- Luong attention for context-aware decoding
- SentencePiece BPE tokenization for both source and target languages
- Greedy decoding for fast inference and beam search for better quality
- BLEU and BERTScore evaluation metrics
- Teacher forcing during training for stable convergence
- Support for both local training and Google Colab

## Model Architecture

### Encoder
- **Type**: Bidirectional LSTM
- **Embedding Dimension**: 256
- **Hidden Size**: 512
- **Number of Layers**: 2
- **Dropout**: 0.3
- **Encoder Output Size**: 1024 (2 × 512, bidirectional)

### Decoder
- **Type**: LSTM with Luong Attention
- **Embedding Dimension**: 256
- **Hidden Size**: 1024
- **Number of Layers**: 1
- **Attention Type**: Luong Attention
- **Dropout**: 0.3

### Tokenization
- **Source Vocabulary Size**: 8,000 (SentencePiece BPE)
- **Target Vocabulary Size**: 5,000 (SentencePiece BPE)
- **Maximum Sequence Length**: 50 tokens
- **Special Tokens**: pad_id=0, unk_id=1, bos_id=2, eos_id=3

## Requirements

### Python Packages
```bash
pip install torch pandas nltk sentencepiece bert-score
```

**Version Info (Recommended):**
- Python >= 3.8
- PyTorch >= 1.9.0
- sentencepiece >= 0.1.96
- bert-score >= 0.3.12
- nltk >= 3.6
- pandas >= 1.3.0
- numpy >= 1.19.0

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (highly recommended)
  - Minimum: 2GB VRAM
  - Recommended: 4GB+ VRAM
- **CPU**: Can run on CPU but will be significantly slower (5-10x slower than GPU)
- **RAM**: 8GB+ recommended

## Dataset Setup

### Local Setup

1. **Download Dataset Files**
   Create a `data` folder in the project root:
   ```bash
   mkdir -p data
   ```

2. **Add CSV Files**
   Place the following CSV files in the `data/` folder:
   ```
   data/
   ├── train_sa_10000.csv       # Training Sanskrit sentences
   ├── train_en_10000.csv       # Training English translations
   ├── dev_sa_1000.csv          # Validation Sanskrit sentences
   ├── dev_en_1000.csv          # Validation English translations
   ├── test_sa_1000.csv         # Test Sanskrit sentences
   └── test_en_1000.csv         # Test English translations
   ```

   **CSV Format:**
   - Should have a `Source_id` column (unique identifier)
   - `Sentence_sa`: Sanskrit sentences (for source files)
   - `Sentence_en`: English sentences (for target files)

### Google Colab Setup

1. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Create data folder in Colab:**
   ```python
   import os
   os.makedirs('/content/drive/MyDrive/content/data', exist_ok=True)
   ```

3. **Upload CSV Files:**
   - In Google Drive, navigate to `/MyDrive/content/data/`
   - Upload all 6 CSV files:
     - `train_sa_10000.csv`
     - `train_en_10000.csv`
     - `dev_sa_1000.csv`
     - `dev_en_1000.csv`
     - `test_sa_1000.csv`
     - `test_en_1000.csv`

4. **Update data paths in notebook:**
   ```python
   # At the beginning of your notebook, add:
   import os
   os.chdir('/content/drive/MyDrive/content')
   
   # Now run the rest of the script
   ```

## How to Run

### Option 1: Local Training (Python Script)

1. **Install dependencies:**
   ```bash
   pip install torch pandas nltk sentencepiece bert-score
   ```

2. **Ensure dataset is in place:**
   ```bash
   ls data/
   # Should show: train_sa_10000.csv, train_en_10000.csv, etc.
   ```

3. **Run training (via Jupyter or by converting notebook to script):**
   ```bash
   jupyter notebook sanskrit_mnt_minimal_code.ipynb
   ```
   
   Or convert and run as Python:
   ```bash
   python sanskrit_mnt_minimal_code.py
   ```

   The script will:
   - Load and process datasets
   - Train SentencePiece tokenizers
   - Initialize the LSTM encoder-decoder model
   - Train for 20 epochs with early stopping based on BLEU
   - Save best model to `best_model.pt`
   - Generate `submission.csv` with test predictions
   - Calculate BLEU and BERTScore metrics

### Option 2: Jupyter Notebook in Google Colab (Recommended for GPU)

1. **Upload notebook to Colab:**
   - Go to [Google Colab](https://colab.research.google.com)
   - Click "Upload" and select `sanskrit_mnt_minimal_code.ipynb`

2. **Mount Google Drive and set paths:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   import os
   os.chdir('/content/drive/MyDrive/content')
   ```

3. **Enable GPU (Important!):**
   - Click menu: `Runtime` → `Change runtime type`
   - Select `GPU` under "Hardware accelerator"
   - Click `Save`

4. **Install packages:**
   ```python
   !pip install torch pandas nltk sentencepiece bert-score
   ```

5. **Run all cells:**
   - Cell-by-cell execution (Shift+Enter)
   - Or Ctrl+F9 to run all cells at once

### Option 3: Command-line with GPU Check

```bash
# Ensure CUDA is available
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"

# Run training (requires converting notebook to script)
python sanskrit_mnt_minimal_code.py
```

## Training Details

### Training Configuration
```
Batch Size: 32
Learning Rate: 1e-3 (0.001)
Optimizer: Adam
LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
Teacher Forcing Ratio: 0.5
Loss Function: CrossEntropyLoss (ignore padding)
Number of Epochs: 20
```

### Training Process
```
Epoch 1: Dataset Loading → Tokenizer Training → Model Init → Training Loop
         └─ Prints: Train Loss, Dev BLEU
Epoch 2-20: Training → Evaluation → Check Best Model
         └─ Best model saved when dev BLEU improves
         └─ LR scheduler adjusts learning rate if BLEU plateaus
```

### Output Files Generated
- `spm_source.model` - SentencePiece model for Sanskrit (vocab: 8,000)
- `spm_target.model` - SentencePiece model for English (vocab: 5,000)
- `best_model.pt` - Best model weights (saved when dev BLEU improves)
- `submission.csv` - Test set predictions with Source_id and translations

## Decoding Methods

### Greedy Decoding (Fast, Used for Validation)
- Selects the highest probability token at each step
- Speed: ~100-200 samples/sec on GPU
- Quality: Good for quick validation
- Used during training evaluation

### Beam Search (Higher Quality, Used for Test)
- Maintains multiple hypotheses (beam_size=4)
- Explores multiple paths and selects the best
- Speed: ~10-30 samples/sec on GPU
- Quality: Better BLEU scores
- Used for final test evaluation

## Evaluation Metrics

### BLEU Score
- Measures n-gram overlap between predictions and references
- Scale: 0-1 (1.0 = perfect match)
- Typical range for this model: 0.10-0.40 (on small dataset)
- Calculated using smoothing function (method1)

### BERTScore F1
- Semantic similarity using BERT embeddings
- Rescaled with baseline
- Computed on test set
- Range: -1 to 1

## Performance Notes

### Training Time
- **Per Epoch**: ~1-3 minutes on GPU (Tesla T4+), ~5-10 minutes on CPU
- **Total Training**: 20 epochs ~ 30-60 minutes on GPU, ~2+ hours on CPU

### Inference Speed
- **Greedy Decoding (batch_size=32, GPU)**: ~200 samples/min
- **Beam Search (batch_size=32, GPU, beam_size=4)**: ~30 samples/min
- **CPU**: 5-10x slower than GPU

### Memory Requirements
- **GPU VRAM**: ~2-3 GB (with batch_size=32)
- **Disk**: ~50-100 MB (model + tokenizers)
- **RAM**: ~8 GB recommended

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```python
# Reduce batch size
batch_size = 16  # instead of 32

# Or reduce model size
hidden_size = 256  # instead of 512
embed_size = 128   # instead of 256
```

### Issue: "No module named 'sentencepiece'"
**Solution:**
```bash
pip install sentencepiece
```

### Issue: "Dataset file not found"
**Solution:**
- Verify CSV files are in `data/` folder
- Check file names match exactly: `train_sa_10000.csv`, etc.
- For Colab: confirm files are uploaded to `/content/drive/MyDrive/content/data/`

### Issue: "Training loss not decreasing"
**Solution:**
- Check GPU is being used: `torch.cuda.is_available()` should be `True`
- Verify data is loading correctly
- Try increasing learning rate: `lr=5e-3` instead of `1e-3`
- Check teacher forcing ratio: increase it (e.g., 0.7 instead of 0.5)

### Issue: "BLEU score is very low (< 0.05)"
**Possible causes:**
- Model is still training (normal for early epochs)
- Learning rate might need adjustment
- Data format issues (verify CSV column names)
- Model might need more training (20+ epochs)
- Try increasing embedding size or hidden size

## Model Architecture Details

### Why LSTM + Attention?
- LSTM captures sequential dependencies in source and target
- Bidirectional LSTM in encoder captures context from both directions
- Luong attention allows decoder to focus on relevant source tokens
- Teacher forcing stabilizes training

### Key Design Choices
- **Bidirectional Encoder**: Better context understanding of source language
- **Luong Attention**: Efficient attention mechanism (multiplicative)
- **Teacher Forcing**: Improves training stability and convergence
- **ReduceLROnPlateau**: Adapts learning rate based on validation performance

## References

- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762) - Original attention paper
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) - Luong Attention
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) - Original Seq2Seq
- [SentencePiece](https://github.com/google/sentencepiece) - Tokenization library
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NLTK BLEU Score](https://www.nltk.org/api/nltk.translate.html)
- [BERTScore](https://github.com/Tiiiger/bert_score)

## License

[Specify your license here, e.g., MIT, Apache 2.0]

## Author

Aashique Karn

## Contact

For issues or questions, please open an issue on the project repository.

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (highly recommended)
  - Minimum: 2GB VRAM
  - Recommended: 4GB+ VRAM
- **CPU**: Can run on CPU but will be significantly slower (10-100x slower than GPU)
- **RAM**: 8GB+ recommended

## Dataset Setup

### Local Setup

1. **Download Dataset Files**
   Create a `data` folder in the project root:
   ```bash
   mkdir -p data
   ```

2. **Add CSV Files**
   Place the following CSV files in the `data/` folder:
   ```
   data/
   ├── train_sa_10000.csv       # Training Sanskrit sentences
   ├── train_en_10000.csv       # Training English translations
   ├── dev_sa_1000.csv          # Validation Sanskrit sentences
   ├── dev_en_1000.csv          # Validation English translations
   ├── test_sa_1000.csv         # Test Sanskrit sentences
   └── test_en_1000.csv         # Test English translations
   ```

   **CSV Format:**
   - Should have a `Source_id` column (unique identifier)
   - `Sentence_sa`: Sanskrit sentences (for source files)
   - `Sentence_en`: English sentences (for target files)

### Google Colab Setup

1. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Create data folder in Colab:**
   ```python
   import os
   os.makedirs('/content/drive/MyDrive/content/data', exist_ok=True)
   ```

3. **Upload CSV Files:**
   - In Google Drive, navigate to `/MyDrive/content/data/`
   - Upload all 6 CSV files:
     - `train_sa_10000.csv`
     - `train_en_10000.csv`
     - `dev_sa_1000.csv`
     - `dev_en_1000.csv`
     - `test_sa_1000.csv`
     - `test_en_1000.csv`

4. **Update data paths in notebook:**
   ```python
   # At the beginning of your notebook, add:
   import os
   os.chdir('/content/drive/MyDrive/content')
   
   # Now run the rest of the script
   ```

## How to Run

### Option 1: Local Training (Python Script)

1. **Install dependencies:**
   ```bash
   pip install torch torchtext sentencepiece nltk bert-score pandas numpy matplotlib
   ```

2. **Ensure dataset is in place:**
   ```bash
   ls data/
   # Should show: train_sa_10000.csv, train_en_10000.csv, etc.
   ```

3. **Run training:**
   ```bash
   python sanskrit_mnt_minimal_code.py
   ```

   The script will:
   - Load and process datasets
   - Train SentencePiece tokenizers
   - Initialize the Transformer model
   - Train for up to 30 epochs (with early stopping)
   - Save best model to `best_model.pt`
   - Evaluate on test set
   - Generate `submission.csv`

### Option 2: Jupyter Notebook in Google Colab (Recommended for GPU)

1. **Upload notebook to Colab:**
   - Go to [Google Colab](https://colab.research.google.com)
   - Click "Upload" and select `sanskrit_mnt_minimal_code.ipynb`

2. **Mount Google Drive and set paths:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   import os
   os.chdir('/content/drive/MyDrive/content')
   ```

3. **Enable GPU (Important!):**
   - Click menu: `Runtime` → `Change runtime type`
   - Select `GPU` under "Hardware accelerator"
   - Click `Save`

4. **Install packages:**
   ```python
   !pip install torch torchtext sentencepiece nltk bert-score pandas numpy matplotlib
   ```

5. **Run all cells:**
   - Cell-by-cell execution (Shift+Enter)
   - Or Ctrl+F9 to run all cells at once

### Option 3: Command-line Python on GPU (with explicit device)

```bash
# Ensure CUDA is available
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"

# Run training
python sanskrit_mnt_minimal_code.py
```

## Training Details

### Training Process
```
Epoch 1: Dataset Loading → Tokenizer Training → Model Init → Training Loop
         └─ Prints: Train Loss, Dev BLEU, Sample Predictions
Epoch 2-30: Training → Evaluation → Check Early Stopping
         └─ Best model saved when dev BLEU improves
```

### Output Files Generated
- `spm_source.model` - SentencePiece model for Sanskrit
- `spm_source.vocab` - SentencePiece vocabulary for Sanskrit
- `spm_target.model` - SentencePiece model for English
- `spm_target.vocab` - SentencePiece vocabulary for English
- `best_model.pt` - Best model weights (saved when dev BLEU improves)
- `submission.csv` - Test set predictions with Source_id and translations



## Decoding Methods

### Greedy Decoding (Fast, Used for Validation)
- Selects the highest probability token at each step
- Speed: ~100-200 samples/sec on GPU
- Quality: Good for quick validation
- Used during training evaluation

### Beam Search (Higher Quality, Used for Test)
- Maintains multiple hypotheses (beam_size=4)
- Explores multiple paths and selects the best
- Speed: ~10-30 samples/sec on GPU
- Quality: Better BLEU scores
- Used for final test evaluation

## Evaluation Metrics

### BLEU Score
- Measures n-gram overlap between predictions and references
- Scale: 0-1 (1.0 = perfect match)
- Typical range for this model: 0.10-0.40 (on small dataset)
- Calculated using smoothing function (method1)

### BERTScore F1
- Semantic similarity using BERT embeddings
- Rescaled with baseline
- Computed on test set
- Range: -1 to 1

## Performance Notes

### Training Time
- **Per Epoch**: ~1-3 minutes on GPU (Tesla T4+), ~5-10 minutes on CPU
- **Total Training**: 20 epochs ~ 30-60 minutes on GPU, ~2+ hours on CPU

### Inference Speed
- **Greedy Decoding (batch_size=32, GPU)**: ~200 samples/min
- **Beam Search (batch_size=32, GPU, beam_size=4)**: ~30 samples/min
- **CPU**: 5-10x slower than GPU

### Memory Requirements
- **GPU VRAM**: ~2-3 GB (with batch_size=32)
- **Disk**: ~50-100 MB (model + tokenizers)
- **RAM**: ~8 GB recommended

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```python
# Reduce batch size
batch_size = 16  # instead of 32

# Or reduce model size
hidden_size = 256  # instead of 512
embed_size = 128   # instead of 256
```

### Issue: "No module named 'sentencepiece'"
**Solution:**
```bash
pip install sentencepiece
```

### Issue: "Dataset file not found"
**Solution:**
- Verify CSV files are in `data/` folder
- Check file names match exactly: `train_sa_10000.csv`, etc.
- For Colab: confirm files are uploaded to `/content/drive/MyDrive/content/data/`

### Issue: "Training loss not decreasing"
**Solution:**
- Check GPU is being used: `torch.cuda.is_available()` should be `True`
- Verify data is loading correctly
- Try increasing learning rate: `lr=5e-3` instead of `1e-3`
- Check teacher forcing ratio: increase it (e.g., 0.7 instead of 0.5)

### Issue: "BLEU score is very low (< 0.05)"
**Possible causes:**
- Model is still training (normal for early epochs)
- Learning rate might need adjustment
- Data format issues (verify CSV column names)
- Model might need more training (20+ epochs)
- Try increasing embedding size or hidden size

## Model Architecture Details

### Why LSTM + Attention?
- LSTM captures sequential dependencies in source and target
- Bidirectional LSTM in encoder captures context from both directions
- Luong attention allows decoder to focus on relevant source tokens
- Teacher forcing stabilizes training

### Key Design Choices
- **Bidirectional Encoder**: Better context understanding of source language
- **Luong Attention**: Efficient attention mechanism (multiplicative)
- **Teacher Forcing**: Improves training stability and convergence
- **ReduceLROnPlateau**: Adapts learning rate based on validation performance

## References

- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762) - Original attention paper
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) - Luong Attention
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) - Original Seq2Seq
- [SentencePiece](https://github.com/google/sentencepiece) - Tokenization library
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NLTK BLEU Score](https://www.nltk.org/api/nltk.translate.html)
- [BERTScore](https://github.com/Tiiiger/bert_score)

## License

[Specify your license here, e.g., MIT, Apache 2.0]

## Author

Aashique Karn

## Contact

For issues or questions, please open an issue on the project repository.
