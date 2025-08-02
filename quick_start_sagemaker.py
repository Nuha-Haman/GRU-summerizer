#!/usr/bin/env python3
"""
Quick Start Script for SageMaker GRU Text Summarizer
Run this script to quickly set up and train your model in SageMaker
"""

import os
import sys
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput
import numpy as np
import json
from datetime import datetime

def setup_sagemaker():
    """Initialize SageMaker session and configuration"""
    print("🚀 Setting up SageMaker GRU Text Summarizer...")
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()
    bucket = sagemaker_session.default_bucket()
    
    print(f"✅ SageMaker Session initialized:")
    print(f"   - Region: {sagemaker_session.boto_region_name}")
    print(f"   - Role: {role}")
    print(f"   - Bucket: {bucket}")
    
    return sagemaker_session, role, bucket

def create_sample_data():
    """Create sample data for testing (replace with your actual data)"""
    print("📊 Creating sample training data...")
    
    # Create sample data structure
    # Replace this with your actual data preparation
    sample_size = 1000
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
    np.savez('train_data.npz', 
             x_tr=x_tr, y_tr=y_tr, decoder_input_tr=decoder_input_tr)
    np.savez('val_data.npz', 
             x_val=x_val, y_val=y_val, decoder_input_val=decoder_input_val)
    
    print("✅ Sample data created")
    return 'train_data.npz', 'val_data.npz'

def upload_data_to_s3(bucket, train_file, val_file):
    """Upload training data to S3"""
    print("📤 Uploading data to S3...")
    
    s3_client = boto3.client('s3')
    
    # Upload training data
    s3_client.upload_file(train_file, bucket, 'gru-summarizer/data/train_data.npz')
    s3_client.upload_file(val_file, bucket, 'gru-summarizer/data/val_data.npz')
    
    print(f"✅ Data uploaded to s3://{bucket}/gru-summarizer/data/")

def create_training_estimator(role, bucket):
    """Create TensorFlow estimator with optimized settings"""
    print("🔧 Creating training estimator...")
    
    estimator = TensorFlow(
        entry_point='train.py',
        role=role,
        instance_count=1,
        instance_type='ml.p3.2xlarge',  # GPU instance for training
        framework_version='2.10.0',
        py_version='py39',
        hyperparameters={
            'max_len_text': 300,
            'max_len_summary': 30,
            'latent_dim': 256,
            'batch_size': 64,
            'epochs': 50,
            'learning_rate': 1e-3
        },
        output_path=f's3://{bucket}/gru-summarizer/output',
        code_location=f's3://{bucket}/gru-summarizer/code',
        use_spot_instances=True,  # Cost optimization
        max_wait=3600,  # Maximum wait time for spot instances
        max_run=7200,   # Maximum training time (2 hours)
        debugger_hook_config=False,  # Disable debugger for faster training
        disable_profiler=True,  # Disable profiler for faster training
    )
    
    print("✅ Training estimator created")
    return estimator

def start_training(estimator, bucket):
    """Start the training job"""
    print("🏃 Starting training job...")
    
    # Prepare training input
    training_input = TrainingInput(
        s3_data=f's3://{bucket}/gru-summarizer/data',
        content_type='application/x-npy'
    )
    
    # Start training
    estimator.fit({'training': training_input})
    
    print("✅ Training completed!")
    return estimator

def deploy_model(estimator):
    """Deploy the trained model to an endpoint"""
    print("🚀 Deploying model to endpoint...")
    
    endpoint_name = f'gru-summarizer-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',  # CPU instance for inference
        endpoint_name=endpoint_name,
        accelerator_type=None,  # No GPU for inference to save costs
        wait=True
    )
    
    print(f"✅ Model deployed to endpoint: {endpoint_name}")
    return predictor, endpoint_name

def test_endpoint(predictor):
    """Test the deployed endpoint"""
    print("🧪 Testing endpoint...")
    
    test_data = {
        'text': 'This is a sample text for testing the GRU summarizer. The model should generate a summary of this text.'
    }
    
    try:
        result = predictor.predict(test_data)
        print(f"✅ Prediction successful: {result}")
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        print("Note: This might be expected if the inference script needs customization")

def cleanup(predictor, endpoint_name):
    """Clean up resources to avoid charges"""
    print("🧹 Cleaning up resources...")
    
    try:
        predictor.delete_endpoint()
        print(f"✅ Endpoint {endpoint_name} deleted")
    except Exception as e:
        print(f"⚠️  Warning: Could not delete endpoint: {e}")
        print("Please manually delete the endpoint to avoid charges")

def main():
    """Main function to run the complete pipeline"""
    print("=" * 60)
    print("🚀 SageMaker GRU Text Summarizer - Quick Start")
    print("=" * 60)
    
    try:
        # Step 1: Setup SageMaker
        sagemaker_session, role, bucket = setup_sagemaker()
        
        # Step 2: Create sample data (replace with your actual data)
        train_file, val_file = create_sample_data()
        
        # Step 3: Upload data to S3
        upload_data_to_s3(bucket, train_file, val_file)
        
        # Step 4: Create training estimator
        estimator = create_training_estimator(role, bucket)
        
        # Step 5: Start training
        estimator = start_training(estimator, bucket)
        
        # Step 6: Deploy model
        predictor, endpoint_name = deploy_model(estimator)
        
        # Step 7: Test endpoint
        test_endpoint(predictor)
        
        print("\n" + "=" * 60)
        print("🎉 Setup completed successfully!")
        print("=" * 60)
        print(f"📊 Training job: {estimator.latest_training_job.name}")
        print(f"🔗 Endpoint: {endpoint_name}")
        print(f"💰 Estimated cost: ~$5-10 for training, ~$0.50/hour for endpoint")
        print("\n📝 Next steps:")
        print("1. Customize the training script for your specific data")
        print("2. Implement proper inference logic in inference.py")
        print("3. Set up monitoring and auto-scaling")
        print("4. Delete the endpoint when not in use to save costs")
        
        # Ask user if they want to clean up
        response = input("\n❓ Do you want to delete the endpoint now? (y/n): ")
        if response.lower() == 'y':
            cleanup(predictor, endpoint_name)
        else:
            print(f"⚠️  Remember to delete endpoint {endpoint_name} manually to avoid charges")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        print("Please check your AWS credentials and permissions")
        sys.exit(1)

if __name__ == "__main__":
    main() 