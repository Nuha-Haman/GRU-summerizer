#!/usr/bin/env python3
"""
BiGRU Text Summarizer - SageMaker Minimal Cost Runner (No Deployment)
Train and evaluate BiGRU model with ROUGE evaluation and summary generation.
"""

import os
import sys
import json
import numpy as np
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput
from datetime import datetime

def setup_sagemaker():
    """Initialize SageMaker session"""
    print("🚀 Setting up SageMaker...")
    
    session = sagemaker.Session()
    role = get_execution_role()
    bucket = "new-lexi"
    
    print(f"✅ SageMaker ready:")
    print(f"   - Bucket: {bucket}")
    print(f"   - Role: {role}")
    print(f"   - Region: {session.boto_region_name}")
    
    return session, role, bucket

def create_training_script():
    """Create the training script with BiGRU and ROUGE evaluation"""
    script = '''
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout, TimeDistributed, BatchNormalization, Layer, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
from rouge_score import rouge_scorer
import pickle

tf.config.run_functions_eagerly(True)

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        encoder_shape, decoder_shape = input_shape

        self.W_a = self.add_weight(name='W_a',
                                   shape=(encoder_shape[2], encoder_shape[2]),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=(decoder_shape[2], encoder_shape[2]),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=(encoder_shape[2], 1),
                                   initializer='glorot_uniform',
                                   trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        encoder_outputs, decoder_outputs = inputs
        decoder_expanded = tf.expand_dims(decoder_outputs, 2)
        encoder_expanded = tf.expand_dims(encoder_outputs, 1)

        score = K.tanh(
            tf.linalg.matmul(encoder_expanded, self.W_a) + tf.linalg.matmul(decoder_expanded, self.U_a)
        )
        score = tf.linalg.matmul(score, self.V_a)
        score = tf.squeeze(score, axis=-1)
        attention_weights = tf.nn.softmax(score, axis=-1)

        context = tf.matmul(attention_weights, encoder_outputs)
        return context, attention_weights

    def compute_output_shape(self, input_shape):
        encoder_shape, decoder_shape = input_shape
        return [
            tf.TensorShape((decoder_shape[0], decoder_shape[1], encoder_shape[2])),
            tf.TensorShape((decoder_shape[0], decoder_shape[1], encoder_shape[1]))
        ]

def load_data(data_dir):
    """Load training, validation, and test data"""
    print("📊 Loading data...")
    train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    val_data = np.load(os.path.join(data_dir, 'val_data.npz'))
    test_data = np.load(os.path.join(data_dir, 'test_data.npz'))
    
    print(f"✅ Data loaded:")
    print(f"   - Training samples: {len(train_data['x_tr'])}")
    print(f"   - Validation samples: {len(val_data['x_val'])}")
    print(f"   - Test samples: {len(test_data['x_test'])}")
    
    return (train_data['x_tr'], train_data['y_tr'], train_data['decoder_input_tr']), \
           (val_data['x_val'], val_data['y_val'], val_data['decoder_input_val']), \
           (test_data['x_test'], test_data['y_test'], test_data['decoder_input_test'])

def create_bigru_model(max_len_text, y_voc_size, latent_dim=256, embedding_dim=300):
    """Create BiGRU model with attention"""
    x_voc_size = 70000  # Will be updated based on actual data
    
    print("🔧 Building BiGRU model with attention...")
    print(f"   - Max text length: {max_len_text}")
    print(f"   - Output vocab size: {y_voc_size}")
    print(f"   - Latent dim: {latent_dim}")
    print(f"   - Embedding dim: {embedding_dim}")
    
    # Encoder
    encoder_input = Input(shape=(max_len_text,), name='encoder_input', dtype='int32')
    enc_emb = Embedding(x_voc_size, embedding_dim, trainable=True)(encoder_input)
    
    # Bidirectional GRU encoder
    encoder_bigru = Bidirectional(
        GRU(latent_dim, return_sequences=True, return_state=True, dropout=0.3),
        merge_mode='concat'
    )
    encoder_outputs, forward_h, backward_h = encoder_bigru(enc_emb)
    state_h = Concatenate()([forward_h, backward_h])
    
    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    dec_emb = Embedding(y_voc_size, embedding_dim, trainable=True)(decoder_inputs)
    
    # Decoder GRU (twice the latent_dim to match encoder output)
    decoder_gru = GRU(latent_dim*2, return_sequences=True, return_state=True, dropout=0.3)
    decoder_outputs, _ = decoder_gru(dec_emb, initial_state=[state_h])
    
    # Attention mechanism
    attn_layer = AttentionLayer()
    attn_out, _ = attn_layer([encoder_outputs, decoder_outputs])
    
    # Concatenate decoder outputs with attention context
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])
    
    # Output layer
    decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))(decoder_concat_input)
    
    model = Model([encoder_input, decoder_inputs], decoder_dense)
    model.summary()
    return model

def greedy_decode(model, input_seq, tokenizer_summary, max_len_summary=30):
    """Generate summary using greedy decoding"""
    reverse_target_index = {i: w for w, i in tokenizer_summary.word_index.items()}
    
    decoder_input = np.zeros((1, max_len_summary))
    decoder_input[0, 0] = tokenizer_summary.word_index['<sos>']
    
    summary = []
    for i in range(1, max_len_summary):
        preds = model.predict([input_seq, decoder_input], verbose=0)
        pred_id = np.argmax(preds[0, i-1])
        if reverse_target_index.get(pred_id) == '<eos>':
            break
        summary.append(reverse_target_index.get(pred_id, ''))
        decoder_input[0, i] = pred_id
    
    return ' '.join(summary)

def train_and_evaluate():
    """Main training and evaluation function"""
    print("🚀 Starting BiGRU model training and evaluation...")
    
    hyperparameters = json.loads(os.environ.get('SM_HYPERPARAMETERS', '{}'))
    max_len_text = hyperparameters.get('max_len_text', 300)
    latent_dim = hyperparameters.get('latent_dim', 256)
    batch_size = hyperparameters.get('batch_size', 32)
    epochs = hyperparameters.get('epochs', 10)
    y_voc_size = hyperparameters.get('y_voc_size', 5000)
    embedding_dim = hyperparameters.get('embedding_dim', 300)
    
    # Load data
    train_data, val_data, test_data = load_data(os.environ['SM_CHANNEL_TRAINING'])
    
    # Create model
    model = create_bigru_model(max_len_text, y_voc_size, latent_dim, embedding_dim)
    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    checkpoint_path = os.path.join(os.environ['SM_MODEL_DIR'], 'model_checkpoint.weights.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
    ]
    
    # Train model
    model.fit(
        {'encoder_input': train_data[0], 'decoder_input': train_data[2]},
        train_data[1],
        validation_data=({'encoder_input': val_data[0], 'decoder_input': val_data[2]}, val_data[1]),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    # Save the final model
    save_path = os.path.join(os.environ.get('SM_MODEL_DIR', './model'))
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path, save_format='tf')
    print(f"✅ Model saved to: {save_path}")
    
    # Load best weights for evaluation
    model.load_weights(checkpoint_path)
    
    # Load tokenizers
    with open(os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'summary_tokenizer.pkl'), 'rb') as f:
        summary_tokenizer = pickle.load(f)
    
    # ROUGE evaluation
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = []
    
    print("📊 Running ROUGE evaluation on test set...")
    for i in range(100):  # Evaluate on 100 test samples
        article_seq = test_data[0][i:i+1]
        ref_summary = test_data[1][i]
        gen_summary = greedy_decode(model, article_seq, summary_tokenizer)
        
        # Convert reference summary from token IDs to text
        ref_text = ' '.join([str(w) for w in ref_summary if w != 0])  # 0 is padding token
        
        score = scorer.score(ref_text, gen_summary)
        scores.append(score)
    
    # Calculate average ROUGE scores
    avg_rouge = {k: np.mean([s[k].fmeasure for s in scores]) for k in scores[0]}
    print(f"\n📊 Average ROUGE Scores:")
    for metric, score in avg_rouge.items():
        print(f"   - {metric}: {score:.4f}")
    
    # Generate sample summaries
    print("\n📝 Sample Summaries:")
    for i in range(5):
        article_seq = test_data[0][i:i+1]
        gen_summary = greedy_decode(model, article_seq, summary_tokenizer)
        print(f"Article {i+1} Summary: {gen_summary}")

if __name__ == "__main__":
    train_and_evaluate()
'''
    
    with open('train.py', 'w') as f:
        f.write(script)
    
    print("✅ Training script created (BiGRU + ROUGE + Summaries)")

def create_sample_data():
    """Create sample data processing function with column rename"""
    print("📊 Creating sample data processing function...")
    
    script = '''
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle

def clean_text(text):
    """Clean text data"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def add_sos_eos(sequences):
    """Add start and end tokens to summaries"""
    return ['<sos> ' + seq + ' <eos>' for seq in sequences]

def prepare_data():
    """Prepare and save training data"""
    # Load data
    df = pd.read_csv('BBCNewsData.csv')
    
    # Rename 'text' column to 'article' if it exists
    if 'text' in df.columns and 'article' not in df.columns:
        df = df.rename(columns={'text': 'article'})
        print("✅ Renamed 'text' column to 'article'")
    
    # Clean data
    df['article'] = df['article'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_text)
    
    # Remove empty entries
    df = df[df['article'].str.len() > 50]
    df = df[df['summary'].str.len() > 10]
    
    # Split data
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Tokenize text data
    max_len_text = 300
    max_len_summary = 30
    
    # Tokenize articles (input)
    article_tokenizer = Tokenizer()
    article_tokenizer.fit_on_texts(train_data['article'])
    
    # Tokenize summaries (output) with special tokens
    summary_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
    train_summaries_with_tokens = add_sos_eos(train_data['summary'])
    summary_tokenizer.fit_on_texts(train_summaries_with_tokens)
    
    # Convert to sequences and pad
    x_tr = pad_sequences(
        article_tokenizer.texts_to_sequences(train_data['article']),
        maxlen=max_len_text, padding='post'
    )
    x_val = pad_sequences(
        article_tokenizer.texts_to_sequences(val_data['article']),
        maxlen=max_len_text, padding='post'
    )
    x_test = pad_sequences(
        article_tokenizer.texts_to_sequences(test_data['article']),
        maxlen=max_len_text, padding='post'
    )
    
    y_tr = pad_sequences(
        summary_tokenizer.texts_to_sequences(add_sos_eos(train_data['summary'])),
        maxlen=max_len_summary, padding='post'
    )
    y_val = pad_sequences(
        summary_tokenizer.texts_to_sequences(add_sos_eos(val_data['summary'])),
        maxlen=max_len_summary, padding='post'
    )
    y_test = pad_sequences(
        summary_tokenizer.texts_to_sequences(add_sos_eos(test_data['summary'])),
        maxlen=max_len_summary, padding='post'
    )
    
    # Create decoder inputs (shifted by 1)
    decoder_input_tr = y_tr[:, :-1]
    decoder_input_val = y_val[:, :-1]
    decoder_input_test = y_test[:, :-1]
    
    # Adjust target outputs (remove first token)
    y_tr_out = y_tr[:, 1:]
    y_val_out = y_val[:, 1:]
    y_test_out = y_test[:, 1:]
    
    # Save data
    np.savez('train_data.npz', x_tr=x_tr, y_tr=y_tr_out, decoder_input_tr=decoder_input_tr)
    np.savez('val_data.npz', x_val=x_val, y_val=y_val_out, decoder_input_val=decoder_input_val)
    np.savez('test_data.npz', x_test=x_test, y_test=y_test_out, decoder_input_test=decoder_input_test)
    
    # Save tokenizers
    with open('summary_tokenizer.pkl', 'wb') as f:
        pickle.dump(summary_tokenizer, f)
    
    print("✅ Data preparation complete")

if __name__ == "__main__":
    prepare_data()
'''
    
    with open('prepare_data.py', 'w') as f:
        f.write(script)
    
    print("✅ Data preparation script created (with column rename step)")

def upload_data_to_s3(bucket):
    """Upload data to S3 bucket"""
    print("📤 Uploading data to S3...")
    
    s3_client = boto3.client('s3')
    
    # Upload training script
    s3_client.upload_file('train.py', bucket, 'bigru-summarizer/code/train.py')
    
    # Upload data preparation script
    s3_client.upload_file('prepare_data.py', bucket, 'bigru-summarizer/code/prepare_data.py')
    
    print(f"✅ Scripts uploaded to s3://{bucket}/bigru-summarizer/code/")

def main():
    print("=" * 60)
    print("🚀 BiGRU Text Summarizer - SageMaker Minimal Cost (No Deployment)")
    print("=" * 60)
    
    try:
        # Setup SageMaker
        session, role, bucket = setup_sagemaker()
        
        # Create training script with BiGRU and ROUGE evaluation
        create_training_script()
        
        # Create data preparation script with column rename
        create_sample_data()
        
        # Upload scripts to S3
        upload_data_to_s3(bucket)
        
        print("\n⚠️ Manual Steps Required:")
        print("1. Upload your BBCNewsData.csv to s3://{bucket}/bigru-summarizer/data/")
        print("2. Run the prepare_data.py script to create train/val/test.npz files")
        print("3. Upload the generated .npz files and tokenizers to s3://{bucket}/bigru-summarizer/data/")
        
        # Create estimator
        estimator = TensorFlow(
            entry_point='train.py',
            role=role,
            instance_count=1,
            instance_type='ml.g4dn.xlarge',  # Cost-effective GPU instance
            framework_version='2.14.1',
            py_version='py310',
            hyperparameters={
                'max_len_text': 300,
                'latent_dim': 256,
                'batch_size': 32,
                'epochs': 10,
                'y_voc_size': 68755,  # Update with your actual vocab size
                'embedding_dim': 300
            },
            output_path=f's3://{bucket}/bigru-summarizer/output',
            use_spot_instances=True,  # For cost savings
            max_wait=3600,  # 1 hour max
            max_run=3600,
            disable_profiler=True
        )
        
        print("\n🔧 Estimator Configuration:")
        print(f"   - Instance: ml.g4dn.xlarge (spot)")
        print(f"   - Framework: TensorFlow 2.14.1")
        print(f"   - Max runtime: 1 hour")
        print(f"   - Output path: s3://{bucket}/bigru-summarizer/output")
        
        # Start training (commented out as we need manual data prep first)
        print("\n⚠️ To start training after preparing data:")
        print(f"estimator.fit({{'training': 's3://{bucket}/bigru-summarizer/data'}})")
        
        print("\n🎉 Setup complete! Follow the manual steps above to proceed with training.")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()