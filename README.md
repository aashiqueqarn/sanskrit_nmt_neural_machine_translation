# Sanskrit Neural Machine Translation (NMT)

A deep learning-based neural machine translation model that converts Sanskrit sentences to English using a Transformer-based sequence-to-sequence architecture.

## Project Overview

This project implements an encoder-decoder Transformer model trained on Sanskrit-English parallel corpus. The model uses SentencePiece tokenization and is optimized for efficient training and inference on GPU.

**Key Features:**
- Transformer-based Seq2Seq architecture (encoder-decoder)
- SentencePiece BPE tokenization for both source and target languages
- Greedy decoding for fast inference and beam search for better quality
- BLEU and BERTScore evaluation metrics
- Early stopping with patience-based model selection
- Support for both local training and Google Colab

## Model Architecture

- **Encoder**: 3-layer Transformer encoder (d_model=256, nhead=4)
- **Decoder**: 3-layer Transformer decoder (d_model=256, nhead=4)
- **Embedding Dimension**: 256
- **Feed-forward Dimension**: 512
- **Attention Heads**: 4
- **Maximum Sequence Length**: 256 tokens
- **Vocabulary Size**: 8,000 (SentencePiece BPE)
- **Total Parameters**: ~3.5M

## Requirements

### Python Packages
```bash
pip install torch torchtext sentencepiece nltk bert-score pandas numpy matplotlib
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
- Beam size = 1
- Selects argmax token at each step
- Speed: ~20-100 samples/sec on GPU
- Quality: Good for quick validation

### Beam Search (Higher Quality, Used for Test)
- Beam size = 4
- Maintains multiple hypotheses
- Speed: ~2-10 samples/sec on GPU
- Quality: Better BLEU scores

## Evaluation Metrics

### BLEU Score
- Measures n-gram overlap between predictions and references
- Scale: 0-1 (1.0 = perfect match)
- Typical range for this model: 0.01-0.15 (small dataset, early stage)

### BERTScore F1
- Semantic similarity using BERT embeddings
- Computed on test set only (time-consuming)
- Range: -1 to 1

## Performance Notes

### Training Time (per epoch)
- **GPU (NVIDIA)**: ~2-5 minutes
- **CPU**: ~20-50 minutes

### Total Training Time
- 30 epochs (with early stopping): ~1-2 hours on GPU, ~10+ hours on CPU

### Inference Speed
- **Greedy (batch_size=16, GPU)**: ~200 samples/min
- **Beam Search (batch_size=16, GPU)**: ~50 samples/min
- **CPU**: 5-10x slower

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```python
# Reduce batch size
batch_size=8  # instead of 16

# Or reduce model size
d_model=128, nhead=2, encoder_layers=2, decoder_layers=2
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
- Verify data is loading correctly (check batch shapes)
- Try increasing learning rate: `lr=5e-4` instead of `1e-4`

### Issue: "BLEU score is 0.0000"
**Possible causes:**
- Model is still training (normal for first epoch)
- Learning rate too high/low (try adjusting)
- Data format issues (verify CSV column names)
- Model not learning to produce EOS tokens


## Key Fixes Applied to Improve Performance

1. **Decoding Bug Fix**: Remove padding tokens before decoding to avoid Unicode replacement characters
2. **Learning Rate Fix**: Disabled per-batch Noam scheduler stepping that caused effective LR to be near-zero
3. **Model Size Optimization**: Reduced from 56M to 3.5M parameters for better convergence on 10k samples
4. **DataLoader Optimization**: Reduced batch size and disabled multi-workers for macOS/Colab compatibility
5. **Faster Validation**: Use greedy decoding for dev evaluation instead of beam search

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
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
