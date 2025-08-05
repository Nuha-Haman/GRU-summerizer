# BBCNews.csv Data Setup Guide

## 🎯 Overview

This guide shows how to use your BBCNews.csv dataset with the GRU text summarizer in SageMaker.

## 📁 Data Requirements

### Expected BBCNews.csv Format:

```csv
# Option 1: With 'article' column
article,summary
"Article text here...","Summary text here..."

# Option 2: With 'text' column (will be renamed automatically)
text,summary
"Article text here...","Summary text here..."
```

### Required Columns:

- `article` OR `text`: The full article text (if `text`, it will be renamed to `article`)
- `summary`: The corresponding summary

## 🚀 Step-by-Step Setup

### Step 1: Upload BBCNews.csv to S3

#### Option A: Using the upload script

```bash
# If you have BBCNews.csv locally
python upload_data_to_s3.py
```

#### Option B: Manual upload

```python
import boto3
import sagemaker

session = sagemaker.Session()
bucket = session.default_bucket()

s3_client = boto3.client('s3')
s3_client.upload_file('BBCNews.csv', bucket, 'BBCNews.csv')
print(f"✅ Uploaded to s3://{bucket}/BBCNews.csv")
```

### Step 2: Run the Updated Script

```bash
python run_sagemaker_gru.py
```

## 📊 What the Script Does with Your Data

### 1. **Data Loading**

- Downloads BBCNews.csv from S3
- Loads and validates the data
- Shows data statistics

### 2. **Data Preprocessing**

- Cleans text (removes special characters, extra spaces)
- Removes empty or very short entries
- Splits data: 70% train, 15% validation, 15% test

### 3. **Tokenization**

- Creates vocabulary from your articles and summaries
- Converts text to numerical sequences
- Pads sequences to fixed lengths

### 4. **Model Training**

- Uses your real vocabulary sizes
- Trains on your actual BBC news data
- Saves the trained model

## 🔧 Data Processing Details

### Text Cleaning:

```python
def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)      # Normalize spaces
    return text.strip()
```

### Data Split:

- **Training**: 70% of data
- **Validation**: 15% of data
- **Test**: 15% of data

### Sequence Processing:

- **Max article length**: 300 tokens
- **Max summary length**: 30 tokens
- **Padding**: Post-padding with zeros

## 📈 Expected Results

### Data Statistics:

```
📊 Loaded 46,897 samples from BBCNews.csv
📋 Columns: ['article', 'summary']
📊 After cleaning: ~45,000 samples
📊 Split: ~31,500 train, ~6,750 val, ~6,750 test
📊 Article vocabulary size: ~50,000 words
📊 Summary vocabulary size: ~15,000 words
```

### Training Configuration:

- **Epochs**: 15 (increased for real data)
- **Batch size**: 32
- **Vocabulary sizes**: Based on your actual data
- **Model**: GRU with attention mechanism

## 🎯 Customization Options

### Adjust Data Processing:

```python
# In run_sagemaker_gru.py, modify these parameters:
max_len_text = 300      # Article length
max_len_summary = 30    # Summary length
test_size = 0.3         # Validation/test split
```

### Adjust Model Parameters:

```python
hyperparameters={
    'max_len_text': 300,
    'max_len_summary': 30,
    'latent_dim': 256,      # GRU units
    'batch_size': 32,       # Adjust based on GPU memory
    'epochs': 15,           # Training epochs
    'y_voc_size': 5000,     # Summary vocabulary
    'x_voc_size': 10000     # Article vocabulary
}
```

## 🚨 Troubleshooting

### Common Issues:

1. **File not found in S3**:

   ```bash
   # Upload the file first
   python upload_data_to_s3.py
   ```

2. **Memory issues**:

   ```python
   # Reduce batch size
   'batch_size': 16
   ```

3. **Training too slow**:

   ```python
   # Reduce data size for testing
   df = df.sample(n=10000, random_state=42)
   ```

4. **Vocabulary too large**:
   ```python
   # Limit vocabulary size
   article_tokenizer = Tokenizer(num_words=10000)
   summary_tokenizer = Tokenizer(num_words=5000)
   ```

## 📊 Data Quality Checks

### Before Training:

```python
# Check data quality
print(f"Total samples: {len(df)}")
print(f"Average article length: {df['article'].str.len().mean():.0f} characters")
print(f"Average summary length: {df['summary'].str.len().mean():.0f} characters")
print(f"Empty articles: {df['article'].isna().sum()}")
print(f"Empty summaries: {df['summary'].isna().sum()}")
```

### After Processing:

```python
# Check processed data
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Article vocabulary: {len(article_tokenizer.word_index) + 1}")
print(f"Summary vocabulary: {len(summary_tokenizer.word_index) + 1}")
```

## 🎉 Success Indicators

### Good Data Quality:

- ✅ Articles > 50 characters
- ✅ Summaries > 10 characters
- ✅ No empty entries
- ✅ Balanced train/val/test split

### Good Training:

- ✅ Loss decreasing
- ✅ Accuracy increasing
- ✅ No overfitting (val loss not increasing)
- ✅ Model saves successfully

## 💰 Cost Optimization

### For Large Datasets:

```python
# Use smaller subset for testing
df = df.sample(n=10000, random_state=42)

# Use smaller model for cost savings
'latent_dim': 128,  # Instead of 256
'batch_size': 16,   # Instead of 32
```

### For Production:

```python
# Use full dataset
# Use larger model
'latent_dim': 512,
'batch_size': 64,
'epochs': 50
```

## 📈 Next Steps

1. **Run the script** with your BBCNews.csv
2. **Monitor training** progress
3. **Test the model** with sample articles
4. **Deploy to endpoint** for production use
5. **Clean up resources** to avoid charges

Your BBCNews.csv data will now be properly processed and used to train a real GRU text summarizer!
