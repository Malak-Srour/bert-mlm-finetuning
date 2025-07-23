# BERT Masked Language Modeling Fine-tuning 🤖

A comprehensive implementation of BERT fine-tuning for Masked Language Modeling (MLM) using the BookCorpus dataset. This project demonstrates how to customize BERT for text prediction tasks with practical examples and interactive predictions.

## 📋 Project Overview

This project fine-tunes a pre-trained BERT model on the BookCorpus dataset to improve its masked language modeling capabilities. The implementation includes custom dataset handling, training loops, and interactive prediction functions that can fill in masked words in sentences.

## ✨ Key Features

- **🔧 Custom MLM Dataset Class** - Handles text masking and tokenization
- **📚 BookCorpus Integration** - Uses streaming dataset loading for memory efficiency  
- **⚡ GPU Training Support** - Automatic device detection and optimization
- **🎯 Interactive Predictions** - Top-k word prediction for masked tokens
- **💾 Model Persistence** - Save and load fine-tuned models
- **📊 Training Monitoring** - Real-time loss tracking and progress updates

## 🛠️ Technology Stack

- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face transformers library
- **Datasets** - Hugging Face datasets library
- **BERT** - Pre-trained bert-base-uncased model
- **BookCorpus** - Large-scale text dataset for training

## 📁 Project Structure

```text
bert-mlm-finetuning/
├── bert_mlm_finetuning.ipynb    # Main Jupyter notebook
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── bert_finetuned/            # Saved model directory (after training)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── outputs/                   # Training outputs and logs
```

## 🚀 Quick Start

### 1. Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/Malak-Srour/bert-mlm-finetuning.git
cd bert-mlm-finetuning

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 2. Run the Notebook

\`\`\`bash
# Start Jupyter notebook
jupyter notebook bert_mlm_finetuning.ipynb
\`\`\`

### 3. Quick Test (After Training)

\`\`\`python
from transformers import BertTokenizer, BertForMaskedLM

# Load your fine-tuned model
model = BertForMaskedLM.from_pretrained('./bert_finetuned')
tokenizer = BertTokenizer.from_pretrained('./bert_finetuned')

# Test prediction
text = "The capital of France is [MASK]."
# Use the prediction functions from the notebook
\`\`\`

## 📖 Implementation Details

### Dataset Preparation
- **BookCorpus Dataset** - Large collection of over 11,000 books
- **Streaming Mode** - Memory-efficient loading for large datasets
- **Custom Masking** - 15% token masking probability (configurable)
- **Tokenization** - BERT-compatible subword tokenization

### Model Architecture
- **Base Model** - bert-base-uncased (110M parameters)
- **Task** - Masked Language Modeling (MLM)
- **Optimizer** - AdamW with 5e-5 learning rate
- **Scheduler** - Linear warmup and decay

### Training Configuration
\`\`\`python
# Key hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_LENGTH = 512
MLM_PROBABILITY = 0.15
EPOCHS = 1  # Adjust based on your needs
\`\`\`

## 🎯 Key Functions

### 1. Text Masking
\`\`\`python
def mask_text(text, tokenizer, mlm_probability=0.3):
    """Applies random masking to create MLM training examples"""
\`\`\`

### 2. Custom Dataset Class
\`\`\`python
class MaskedLanguageModelingDataset(Dataset):
    """Handles text preprocessing and masking for training"""
\`\`\`

### 3. Prediction Functions
\`\`\`python
def predict_top_k_masked_word(text, model, tokenizer, top_k=5):
    """Predicts top-k most likely words for [MASK] tokens"""

def predict_single_masked_word(text, model, tokenizer):
    """Predicts single most likely word for [MASK] tokens"""
\`\`\`

## 📊 Example Results

### Input Examples
\`\`\`
"The quick brown [MASK] jumps over the lazy dog."
"I love to eat [MASK] for breakfast."
"The capital of France is [MASK]."
\`\`\`

### Model Predictions
The fine-tuned model can predict contextually appropriate words:
- **Context-aware** - Considers surrounding words
- **Multiple predictions** - Top-k most likely candidates
- **Confidence scoring** - Probability-based ranking

## 🔧 Customization Options

### Adjust Masking Probability
\`\`\`python
# Change masking percentage
mlm_probability = 0.20  # Mask 20% of tokens instead of 15%
\`\`\`

### Modify Training Parameters
\`\`\`python
# Experiment with different settings
batch_size = 16        # Larger batches (if GPU memory allows)
learning_rate = 2e-5   # Lower learning rate for stability
epochs = 3             # More training epochs
\`\`\`

### Use Different Datasets
\`\`\`python
# Try other text datasets
dataset = load_dataset("wikipedia", "20220301.en", streaming=True)
dataset = load_dataset("openwebtext", streaming=True)
\`\`\`

## 📈 Performance Monitoring

The notebook includes:
- **Real-time loss tracking** during training
- **Batch processing statistics** 
- **Memory usage monitoring**
- **Training time estimation**

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
\`\`\`python
# Reduce batch size
batch_size = 4  # or even 2

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
\`\`\`

**Slow Training**
\`\`\`python
# Use smaller dataset sample
filtered_texts = [next(iter(dataset["train"]))["text"] for _ in range(50)]

# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
\`\`\`

**Dataset Loading Issues**
\`\`\`python
# If BookCorpus fails, try alternative datasets
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", streaming=True)
\`\`\`

## 🚀 Advanced Usage

### Fine-tune on Custom Text
1. Replace BookCorpus with your own text data
2. Adjust masking strategy for domain-specific terms
3. Experiment with different BERT variants (RoBERTa, DistilBERT)

### Production Deployment
1. Export model to ONNX format for faster inference
2. Implement batch prediction for multiple sentences
3. Add caching for frequently predicted patterns

## 📚 Learning Resources

- **BERT Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Transformers Documentation**: [Hugging Face Docs](https://huggingface.co/docs/transformers)
- **MLM Tutorial**: [Masked Language Modeling Guide](https://huggingface.co/course/chapter7/3)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (\`git checkout -b feature/improvement\`)
3. Commit your changes (\`git commit -am 'Add new feature'\`)
4. Push to the branch (\`git push origin feature/improvement\`)
5. Create a Pull Request


## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **BookCorpus** dataset creators
- **PyTorch** team for the deep learning framework
- **BERT** authors for the groundbreaking architecture

## 📞 Contact

- **GitHub**: [@Malak-Srour](https://github.com/Malak-Srour)
- **Email**: malaksrour74@gmail.com
- **LinkedIn**: [LinkedIn](https://www.linkedin.com/in/malak-srour/)

---


