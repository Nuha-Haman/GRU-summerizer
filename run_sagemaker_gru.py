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
    bucket = session.default_bucket()
    
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
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_data(data_dir):
    train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    val_data = np.load(os.path.join(data_dir, 'val_data.npz'))
    return (train_data['x_tr'], train_data['y_tr'], train_data['decoder_input_tr']), \\
           (val_data['x_val'], val_data['y_val'], val_data['decoder_input_val'])

def create_gru_model(max_len_text, y_voc_size, latent_dim=256):
    # Encoder
    encoder_input = Input(shape=(max_len_text,), name='encoder_input')
    enc_emb = Embedding(10000, 300, name="encoder_embedding")(encoder_input)
    
    encoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, dropout=0.3)(enc_emb)
    encoder_outputs, state_h = encoder_gru
    encoder_outputs = BatchNormalization()(encoder_outputs)
    
    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    dec_emb = Embedding(5000, 300, name="decoder_embedding")(decoder_inputs)
    
    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, dropout=0.3)(dec_emb)
    decoder_outputs, _ = decoder_gru(dec_emb, initial_state=state_h)
    decoder_outputs = BatchNormalization()(decoder_outputs)
    
    # Output
    decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))(decoder_outputs)
    
    model = Model(inputs=[encoder_input, decoder_inputs], outputs=decoder_dense)
    return model

def train_model():
    hyperparameters = json.loads(os.environ.get('SM_HYPERPARAMETERS', '{}'))
    
    max_len_text = hyperparameters.get('max_len_text', 300)
    latent_dim = hyperparameters.get('latent_dim', 256)
    batch_size = hyperparameters.get('batch_size', 32)
    epochs = hyperparameters.get('epochs', 5)
    y_voc_size = hyperparameters.get('y_voc_size', 500)
    
    train_data, val_data = load_data(os.environ['SM_CHANNEL_TRAINING'])
    
    model = create_gru_model(max_len_text, y_voc_size, latent_dim)
    
    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint(os.path.join(os.environ['SM_MODEL_DIR'], 'best_model.keras'), 
                       monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        {'encoder_input': train_data[0], 'decoder_input': train_data[2]},
        train_data[1],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            {'encoder_input': val_data[0], 'decoder_input': val_data[2]},
            val_data[1]
        ),
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(os.path.join(os.environ['SM_MODEL_DIR'], 'final_model.keras'))

if __name__ == "__main__":
    train_model()
'''
    
    with open('train.py', 'w') as f:
        f.write(script)
    
    print("✅ Training script created")

def create_sample_data():
    """Create sample data (REPLACE WITH YOUR DATA)"""
    print("📊 Creating sample data...")
    
    sample_size = 500
    max_len_text = 300
    max_len_summary = 30
    
    # Sample data (replace with your actual data)
    x_tr = np.random.randint(0, 1000, (sample_size, max_len_text))
    y_tr = np.random.randint(0, 500, (sample_size, max_len_summary))
    decoder_input_tr = np.random.randint(0, 500, (sample_size, max_len_summary - 1))
    
    x_val = np.random.randint(0, 1000, (sample_size // 5, max_len_text))
    y_val = np.random.randint(0, 500, (sample_size // 5, max_len_summary))
    decoder_input_val = np.random.randint(0, 500, (sample_size // 5, max_len_summary - 1))
    
    # Save data
    np.savez('train_data.npz', x_tr=x_tr, y_tr=y_tr, decoder_input_tr=decoder_input_tr)
    np.savez('val_data.npz', x_val=x_val, y_val=y_val, decoder_input_val=decoder_input_val)
    
    print(f"✅ Sample data created: {sample_size} training samples")
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
        framework_version='2.10.0',
        py_version='py39',
        hyperparameters={
            'max_len_text': 300,
            'max_len_summary': 30,
            'latent_dim': 256,
            'batch_size': 32,
            'epochs': 5,  # Reduced for demo
            'y_voc_size': 500
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