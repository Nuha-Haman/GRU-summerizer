#!/usr/bin/env python3
"""
GRU Text Summarizer - SageMaker Minimal Cost Runner
Run this script to train and deploy your GRU model in SageMaker with minimal cost.
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
    """Create the training script"""
    script = '''
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout, TimeDistributed, BatchNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K

# Enable eager execution to avoid graph mode issues
tf.config.run_functions_eagerly(True)

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
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

        # Expand decoder and encoder dims
        decoder_expanded = tf.expand_dims(decoder_outputs, 2)   # (batch, dec_len, 1, dim)
        encoder_expanded = tf.expand_dims(encoder_outputs, 1)   # (batch, 1, enc_len, dim)

        # Attention score
        score = K.tanh(
            tf.linalg.matmul(encoder_expanded, self.W_a) + tf.linalg.matmul(decoder_expanded, self.U_a)
        )  # (batch, dec_len, enc_len, dim)

        # Apply attention weights
        score = tf.linalg.matmul(score, self.V_a)   # (batch, dec_len, enc_len, 1)
        score = tf.squeeze(score, axis=-1)          # (batch, dec_len, enc_len)
        attention_weights = tf.nn.softmax(score, axis=-1)  # (batch, dec_len, enc_len)

        # Compute context vector
        context = tf.matmul(attention_weights, encoder_outputs)  # (batch, dec_len, dim)

        return context, attention_weights

    def compute_output_shape(self, input_shape):
        encoder_shape, decoder_shape = input_shape
        return [
            tf.TensorShape((decoder_shape[0], decoder_shape[1], encoder_shape[2])),  # context
            tf.TensorShape((decoder_shape[0], decoder_shape[1], encoder_shape[1]))   # attention weights
        ]

def load_data(data_dir):
    """Load training and validation data"""
    try:
        train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
        val_data = np.load(os.path.join(data_dir, 'val_data.npz'))
        
        # Extract data arrays
        x_tr = train_data['x_tr']
        y_tr = train_data['y_tr']
        decoder_input_tr = train_data['decoder_input_tr']
        
        x_val = val_data['x_val']
        y_val = val_data['y_val']
        decoder_input_val = val_data['decoder_input_val']
        
        print(f"📊 Data loaded successfully:")
        print(f"   - Training samples: {len(x_tr)}")
        print(f"   - Validation samples: {len(x_val)}")
        print(f"   - Input shape: {x_tr.shape}")
        print(f"   - Output shape: {y_tr.shape}")
        
        return (x_tr, y_tr, decoder_input_tr), (x_val, y_val, decoder_input_val)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def create_gru_model(max_len_text, y_voc_size, latent_dim=256, embedding_dim=300):
    """Create GRU model with attention for text summarization"""
    # Get vocabulary sizes from hyperparameters
    x_voc_size = 70000  # Will be updated based on actual data
    
    print(f"🔧 Creating GRU model with attention:")
    print(f"   - Max text length: {max_len_text}")
    print(f"   - Output vocab size: {y_voc_size}")
    print(f"   - Latent dim: {latent_dim}")
    print(f"   - Embedding dim: {embedding_dim}")
    
    # Encoder
    encoder_input = Input(shape=(max_len_text,), name='encoder_input', dtype='int32')
    enc_emb = Embedding(x_voc_size, embedding_dim, name="encoder_embedding", trainable=True)(encoder_input)
    
    # Single-layer GRU encoder (simplified to avoid issues)
    encoder_outputs, state_h = GRU(latent_dim, return_sequences=True, return_state=True, dropout=0.3, name="encoder_gru")(enc_emb)
    encoder_outputs = BatchNormalization()(encoder_outputs)
    print("Encoder state shape:", state_h.shape)
    
    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    dec_emb = Embedding(y_voc_size, embedding_dim, name="decoder_embedding", trainable=True)(decoder_inputs)
    
    # Use the same GRU layer for decoder
    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, dropout=0.3, name="decoder_gru")
    
    # decoder_outputs, _ = decoder_gru(dec_emb, initial_state=state_h)
    decoder_outputs, _ = decoder_gru(dec_emb, initial_state=[state_h])

    decoder_outputs = BatchNormalization()(decoder_outputs)
    
    # Attention mechanism
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, _ = attn_layer([encoder_outputs, decoder_outputs])
    
    # Concatenate decoder outputs with attention context
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])
    
    # Output layer
    decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'), name="final_dense")(decoder_concat_input)
    
    model = Model(inputs=[encoder_input, decoder_inputs], outputs=decoder_dense)
    
    print(f"✅ GRU model with attention created successfully")
    model.summary()
    
    return model

def train_model():
    """Main training function with checkpointing"""
    print("🚀 Starting training with checkpoint support...")
    
    # Load hyperparameters
    hyperparameters = json.loads(os.environ.get('SM_HYPERPARAMETERS', '{}'))
    
    max_len_text = hyperparameters.get('max_len_text', 300)
    max_len_summary = hyperparameters.get('max_len_summary', 30)
    latent_dim = hyperparameters.get('latent_dim', 256)
    batch_size = hyperparameters.get('batch_size', 32)
    epochs = hyperparameters.get('epochs', 10)
    y_voc_size = hyperparameters.get('y_voc_size', 5000)
    x_voc_size = hyperparameters.get('x_voc_size', 10000)
    embedding_dim = hyperparameters.get('embedding_dim', 300)
    
    print(f"📊 Hyperparameters:")
    print(f"   - Max text length: {max_len_text}")
    print(f"   - Max summary length: {max_len_summary}")
    print(f"   - Latent dim: {latent_dim}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Y vocab size: {y_voc_size}")
    print(f"   - X vocab size: {x_voc_size}")
    print(f"   - Embedding dim: {embedding_dim}")
    
    # Load data
    train_data, val_data = load_data(os.environ['SM_CHANNEL_TRAINING'])
    
    # Update vocabulary sizes based on actual data
    x_voc_size = max(x_voc_size, np.max(train_data[0]) + 1)
    y_voc_size = max(y_voc_size, np.max(train_data[1]) + 1)
    
    print(f"📊 Final model configuration:")
    print(f"   - Input vocabulary size: {x_voc_size}")
    print(f"   - Output vocabulary size: {y_voc_size}")
    
    # Create model
    model = create_gru_model(max_len_text, y_voc_size, latent_dim, embedding_dim)
    
    # Compile model
    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # === ✅ Add checkpoint support ===
    checkpoint_dir = os.path.join(os.environ['SM_MODEL_DIR'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.weights.h5')

    if os.path.exists(checkpoint_path):
        print(f"📦 Resuming from checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    else:
        print("🆕 No checkpoint found, starting from scratch.")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1)
    ]
    
    # Prepare training and validation data
    train_encoder_input = train_data[0].astype('int32')
    train_decoder_input = train_data[2].astype('int32')
    train_target_output = train_data[1].astype('int32')
    
    val_encoder_input = val_data[0].astype('int32')
    val_decoder_input = val_data[2].astype('int32')
    val_target_output = val_data[1].astype('int32')
    
    print("🏃 Starting model training...")
    
    history = model.fit(
        {'encoder_input': train_encoder_input, 'decoder_input': train_decoder_input},
        train_target_output,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            {'encoder_input': val_encoder_input, 'decoder_input': val_decoder_input},
            val_target_output
        ),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(os.environ['SM_MODEL_DIR'], save_format='tf')
    
    # Save training history
    with open(os.path.join(os.environ['SM_MODEL_DIR'], 'training_history.json'), 'w') as f:
        json.dump(history.history, f)
    
    print("✅ Training completed and model saved!")
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    train_model()
'''
    
    with open('train.py', 'w') as f:
        f.write(script)
    
    print("✅ Training script created")

def create_sample_data():
    """Load and prepare BBCNewsData.csv data from S3"""
    print("📊 Loading BBCNewsData.csv data from S3...")
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import re
    
    # Load data from S3
    s3_client = boto3.client('s3')
    bucket = "new-lexi"
    
    # Try to download the file from S3
    try:
        s3_client.download_file(bucket, 'BBCNewsData.csv', 'BBCNewsData.csv')
        print("✅ Downloaded BBCNewsData.csv from S3")
    except Exception as e:
        print(f"⚠️  Could not download from S3: {e}")
        print("Please ensure BBCNewsData.csv is uploaded to your S3 bucket")
        return None, None
    
    # Load the CSV file
    df = pd.read_csv('BBCNewsData.csv')
    print(f"📊 Loaded {len(df)} samples from BBCNewsData.csv")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Rename 'text' column to 'article' if it exists
    if 'text' in df.columns and 'article' not in df.columns:
        df = df.rename(columns={'text': 'article'})
        print("✅ Renamed 'text' column to 'article'")
    elif 'article' not in df.columns:
        print("❌ Error: No 'article' or 'text' column found in the dataset")
        print(f"Available columns: {list(df.columns)}")
        return None, None
    
    # Clean and prepare data
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Clean the data
    df['article'] = df['article'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_text)
    
    # Remove empty entries
    df = df[df['article'].str.len() > 50]
    df = df[df['summary'].str.len() > 10]
    
    print(f"📊 After cleaning: {len(df)} samples")
    print(f"📋 Final columns: {list(df.columns)}")
    print(f"📊 Sample article length: {df['article'].iloc[0][:100]}...")
    print(f"📊 Sample summary length: {df['summary'].iloc[0][:50]}...")
    
    # Split data
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"📊 Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Tokenize text data
    max_len_text = 300
    max_len_summary = 30
    
    # Tokenize articles (input)
    article_tokenizer = Tokenizer()
    article_tokenizer.fit_on_texts(train_data['article'])
    
    # Tokenize summaries (output) with special tokens
    summary_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
    
    # Add SOS and EOS tokens to summaries
    def add_sos_eos(sequences):
        return ['<sos> ' + seq + ' <eos>' for seq in sequences]
    
    train_summaries_with_tokens = add_sos_eos(train_data['summary'])
    val_summaries_with_tokens = add_sos_eos(val_data['summary'])
    test_summaries_with_tokens = add_sos_eos(test_data['summary'])
    
    summary_tokenizer.fit_on_texts(train_summaries_with_tokens)
    
    # Convert to sequences
    x_tr = article_tokenizer.texts_to_sequences(train_data['article'])
    x_val = article_tokenizer.texts_to_sequences(val_data['article'])
    x_test = article_tokenizer.texts_to_sequences(test_data['article'])
    
    y_tr = summary_tokenizer.texts_to_sequences(train_summaries_with_tokens)
    y_val = summary_tokenizer.texts_to_sequences(val_summaries_with_tokens)
    y_test = summary_tokenizer.texts_to_sequences(test_summaries_with_tokens)
    
    # Pad sequences
    x_tr = pad_sequences(x_tr, maxlen=max_len_text, padding='post')
    x_val = pad_sequences(x_val, maxlen=max_len_text, padding='post')
    x_test = pad_sequences(x_test, maxlen=max_len_text, padding='post')
    
    y_tr = pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
    y_val = pad_sequences(y_val, maxlen=max_len_summary, padding='post')
    y_test = pad_sequences(y_test, maxlen=max_len_summary, padding='post')
    
    # Create decoder inputs (shifted by 1)
    decoder_input_tr = y_tr[:, :-1]
    decoder_input_val = y_val[:, :-1]
    decoder_input_test = y_test[:, :-1]
    
    # Adjust target outputs (remove first token)
    y_tr_out = y_tr[:, 1:]
    y_val_out = y_val[:, 1:]
    y_test_out = y_test[:, 1:]
    
    # Validate shapes
    print(f"📊 Data shape validation:")
    print(f"   - y_tr shape: {y_tr.shape}")
    print(f"   - decoder_input_tr shape: {decoder_input_tr.shape}")
    print(f"   - y_tr_out shape: {y_tr_out.shape}")
    print(f"   - x_tr shape: {x_tr.shape}")
    
    # Ensure all shapes are correct
    assert decoder_input_tr.shape[0] == y_tr_out.shape[0], "Batch sizes don't match"
    assert decoder_input_tr.shape[1] == y_tr_out.shape[1], "Sequence lengths don't match"
    assert x_tr.shape[0] == y_tr_out.shape[0], "Encoder/decoder batch sizes don't match"
    
    # Save processed data
    try:
        np.savez('train_data.npz', 
                 x_tr=x_tr, y_tr=y_tr_out, decoder_input_tr=decoder_input_tr)
        np.savez('val_data.npz', 
                 x_val=x_val, y_val=y_val_out, decoder_input_val=decoder_input_val)
        np.savez('test_data.npz', 
                 x_test=x_test, y_test=y_test_out, decoder_input_test=decoder_input_test)
        
        print("✅ Data files saved successfully")
        
        # Verify saved data
        train_check = np.load('train_data.npz')
        val_check = np.load('val_data.npz')
        
        print(f"📊 Saved data verification:")
        print(f"   - Train data keys: {list(train_check.keys())}")
        print(f"   - Val data keys: {list(val_check.keys())}")
        print(f"   - Train x_tr shape: {train_check['x_tr'].shape}")
        print(f"   - Train y_tr shape: {train_check['y_tr'].shape}")
        print(f"   - Train decoder_input_tr shape: {train_check['decoder_input_tr'].shape}")
        
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        raise
    
    # Save tokenizers for later use
    import pickle
    with open('article_tokenizer.pkl', 'wb') as f:
        pickle.dump(article_tokenizer, f)
    with open('summary_tokenizer.pkl', 'wb') as f:
        pickle.dump(summary_tokenizer, f)
    
    print(f"✅ Data prepared:")
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    print(f"   - Test samples: {len(test_data)}")
    print(f"   - Article vocabulary size: {len(article_tokenizer.word_index) + 1}")
    print(f"   - Summary vocabulary size: {len(summary_tokenizer.word_index) + 1}")
    
    return 'train_data.npz', 'val_data.npz'

def upload_data_to_s3(bucket, train_file, val_file):
    """Upload data to S3"""
    print("📤 Uploading data to S3...")
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(train_file, bucket, 'gru-summarizer/data/train_data.npz')
    s3_client.upload_file(val_file, bucket, 'gru-summarizer/data/val_data.npz')
    
    print(f"✅ Data uploaded to s3://{bucket}/gru-summarizer/data/")

def create_estimator(role, bucket):
    """Create cost-optimized estimator"""
    print("🔧 Creating cost-optimized estimator...")
    
    estimator = TensorFlow(
        entry_point='train.py',
        role=role,
        instance_count=1,
        instance_type='ml.g4dn.xlarge',  # 1 GPU, cheaper than p3
        framework_version='2.14.1',  # Latest supported version
        py_version='py310',
        hyperparameters={
            'max_len_text': 300,
            'max_len_summary': 30,
            'latent_dim': 256,
            'batch_size': 32,
            'epochs': 70,  # Increased for real data
            'y_voc_size': 68755,  # Increased for real vocabulary
            'x_voc_size': 331569,  # Increased for real vocabulary
            'embedding_dim': 300   # FastText embedding dimension
        },
        output_path=f's3://{bucket}/gru-summarizer/output',
        use_spot_instances=True,  # 60-90% cost savings
        max_wait=3600,  # 1 hour max wait
        max_run=3600,   # 1 hour max training
        disable_profiler=True,
    )
    
    print("✅ Estimator created with spot instances")
    return estimator

def start_training(estimator, bucket):
    """Start training job"""
    print("🏃 Starting training...")
    
    training_input = TrainingInput(
        s3_data=f's3://{bucket}/gru-summarizer/data',
        content_type='application/x-npy'
    )
    
    estimator.fit({'training': training_input})
    print("✅ Training completed!")
    return estimator

def deploy_model(estimator):
    """Deploy model to endpoint"""
    print("🚀 Deploying model...")
    
    endpoint_name = f'gru-summarizer-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',  # CPU instance, cost-effective
        endpoint_name=endpoint_name,
        wait=True
    )
    
    print(f"✅ Model deployed: {endpoint_name}")
    print(f"💰 Cost: ~$0.115/hour")
    return predictor, endpoint_name

def test_endpoint(predictor):
    """Test the endpoint"""
    print("🧪 Testing endpoint...")
    
    test_data = {'text': 'This is a test text for the GRU summarizer.'}
    
    try:
        result = predictor.predict(test_data)
        print(f"✅ Test successful: {result}")
    except Exception as e:
        print(f"⚠️  Test failed (expected for demo): {e}")
        print("Customize inference script for your specific model")

def cleanup(predictor, endpoint_name):
    """Clean up resources"""
    print("🧹 Cleaning up...")
    
    try:
        predictor.delete_endpoint()
        print("✅ Endpoint deleted - No more charges!")
    except Exception as e:
        print(f"⚠️  Manual cleanup needed: {e}")
        print(f"Delete endpoint: {endpoint_name}")

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 GRU Text Summarizer - SageMaker Minimal Cost")
    print("=" * 60)
    
    try:
        # Step 1: Setup
        session, role, bucket = setup_sagemaker()
        
        # Step 2: Create training script
        create_training_script()
        
        # Step 3: Create sample data
        train_file, val_file = create_sample_data()
        
        # Step 4: Upload data
        upload_data_to_s3(bucket, train_file, val_file)
        
        # Step 5: Create estimator
        estimator = create_estimator(role, bucket)
        
        # Step 6: Start training
        estimator = start_training(estimator, bucket)
        
        # Step 7: Deploy model
        predictor, endpoint_name = deploy_model(estimator)
        
        # Step 8: Test endpoint
        test_endpoint(predictor)
        
        print("\n" + "=" * 60)
        print("🎉 Setup completed successfully!")
        print("=" * 60)
        print(f"📊 Training job: {estimator.latest_training_job.name}")
        print(f"🔗 Endpoint: {endpoint_name}")
        print(f"💰 Estimated cost: ~$0.60-1.10")
        
        # Ask for cleanup
        response = input("\n❓ Delete endpoint now? (y/n): ")
        if response.lower() == 'y':
            cleanup(predictor, endpoint_name)
        else:
            print(f"⚠️  Remember to delete endpoint {endpoint_name} manually")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 