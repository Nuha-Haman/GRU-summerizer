# SageMaker Best Practices Guide for GRU Text Summarizer

## Overview

This guide provides step-by-step instructions for running the optimized GRU-based text summarizer in AWS SageMaker with industry best practices for cost optimization, performance, and scalability.

## Prerequisites

### 1. AWS Setup

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Install SageMaker SDK
pip install sagemaker
```

### 2. Required Permissions

Ensure your IAM role has the following permissions:

- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`
- `CloudWatchLogsFullAccess`

## Project Structure

```
gru-summarizer/
├── train.py                 # Main training script
├── inference.py             # Inference script for endpoints
├── requirements.txt         # Python dependencies
├── Dockerfile              # Custom container (optional)
├── sagemaker_setup.py      # SageMaker setup utilities
├── sagemaker_training.ipynb # Training notebook
└── data/
    ├── train_data.npz      # Training data
    └── val_data.npz        # Validation data
```

## Step 1: Data Preparation

### 1.1 Prepare Training Data

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your preprocessed data
df = pd.read_csv('your_data.csv')

# Split data
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# Save as numpy arrays
np.savez('data/train_data.npz',
         x_tr=train_data['x_tr'],
         y_tr=train_data['y_tr'],
         decoder_input_tr=train_data['decoder_input_tr'])

np.savez('data/val_data.npz',
         x_val=val_data['x_val'],
         y_val=val_data['y_val'],
         decoder_input_val=val_data['decoder_input_val'])
```

### 1.2 Upload Data to S3

```python
import boto3
import sagemaker

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()

# Upload data to S3
s3_client = boto3.client('s3')
s3_client.upload_file('data/train_data.npz', bucket, 'gru-summarizer/data/train_data.npz')
s3_client.upload_file('data/val_data.npz', bucket, 'gru-summarizer/data/val_data.npz')

print(f"Data uploaded to s3://{bucket}/gru-summarizer/data/")
```

## Step 2: Training Setup

### 2.1 Create Training Estimator

```python
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput

# Create TensorFlow estimator with best practices
estimator = TensorFlow(
    entry_point='train.py',
    role=sagemaker.get_execution_role(),
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
    keep_alive_period_in_seconds=1800,  # Keep instance alive for 30 minutes
)
```

### 2.2 Best Practices for Training

#### Cost Optimization

```python
# Use spot instances for cost savings (up to 90% cheaper)
estimator = TensorFlow(
    # ... other parameters
    use_spot_instances=True,
    max_wait=3600,  # Wait up to 1 hour for spot instances
)

# Use appropriate instance types
# Training: ml.p3.2xlarge (GPU) or ml.g4dn.xlarge (GPU)
# Inference: ml.m5.large (CPU) or ml.c5.large (CPU)
```

#### Performance Optimization

```python
# Enable mixed precision training
mixed_precision.set_global_policy('mixed_float16')

# Use optimal batch size for GPU memory
batch_size = 64  # Adjust based on GPU memory

# Enable gradient clipping
optimizer = AdamW(
    learning_rate=learning_rate,
    weight_decay=1e-5,
    clipnorm=1.0  # Gradient clipping
)
```

## Step 3: Training Execution

### 3.1 Start Training Job

```python
# Prepare training input
training_input = TrainingInput(
    s3_data=f's3://{bucket}/gru-summarizer/data',
    content_type='application/x-npy'
)

# Start training
print("Starting training job...")
estimator.fit({'training': training_input})
print("Training completed!")
```

### 3.2 Monitor Training

```python
# Monitor training progress
import time

while True:
    status = estimator.latest_training_job.describe()['TrainingJobStatus']
    print(f"Training status: {status}")

    if status in ['Completed', 'Failed', 'Stopped']:
        break

    time.sleep(60)  # Check every minute

# Get training metrics
training_job_name = estimator.latest_training_job.name
print(f"Training job name: {training_job_name}")
```

## Step 4: Model Deployment

### 4.1 Deploy to Endpoint

```python
from datetime import datetime

# Deploy model with best practices
endpoint_name = f'gru-summarizer-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',  # CPU instance for inference
    endpoint_name=endpoint_name,
    accelerator_type=None,  # No GPU for inference to save costs
    data_capture_config=None,  # Disable data capture for now
    wait=True
)

print(f"Model deployed to endpoint: {endpoint_name}")
```

### 4.2 Auto Scaling Configuration

```python
# Configure auto scaling for cost optimization
import boto3

autoscaling = boto3.client('application-autoscaling')

# Register scalable target
autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Configure scaling policy
autoscaling.put_scaling_policy(
    PolicyName=f'{endpoint_name}-scaling-policy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,  # Target CPU utilization
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

## Step 5: Testing and Monitoring

### 5.1 Test Endpoint

```python
# Test the endpoint
test_data = {
    'text': 'This is a sample text for testing the GRU summarizer.'
}

try:
    result = predictor.predict(test_data)
    print(f"Prediction: {result}")
except Exception as e:
    print(f"Error testing endpoint: {e}")
```

### 5.2 CloudWatch Monitoring

```python
import boto3

# Setup CloudWatch monitoring
cloudwatch = boto3.client('cloudwatch')

# Create custom metrics
def log_metric(endpoint_name, metric_name, value):
    cloudwatch.put_metric_data(
        Namespace=f'Custom/{endpoint_name}',
        MetricData=[
            {
                'MetricName': metric_name,
                'Value': value,
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name}
                ]
            }
        ]
    )

# Example usage
log_metric(endpoint_name, 'Invocations', 1)
log_metric(endpoint_name, 'ModelLatency', 0.5)
```

## Step 6: Cost Optimization Strategies

### 6.1 Instance Selection

```python
# Training instances (GPU)
training_instances = {
    'small': 'ml.p3.2xlarge',    # 1 GPU, 8 vCPUs, 61 GB RAM
    'medium': 'ml.p3.8xlarge',   # 4 GPUs, 32 vCPUs, 244 GB RAM
    'large': 'ml.p3.16xlarge'    # 8 GPUs, 64 vCPUs, 488 GB RAM
}

# Inference instances (CPU)
inference_instances = {
    'small': 'ml.m5.large',      # 2 vCPUs, 8 GB RAM
    'medium': 'ml.m5.xlarge',    # 4 vCPUs, 16 GB RAM
    'large': 'ml.m5.2xlarge'     # 8 vCPUs, 32 GB RAM
}
```

### 6.2 Spot Instances

```python
# Use spot instances for training (up to 90% cost savings)
estimator = TensorFlow(
    # ... other parameters
    use_spot_instances=True,
    max_wait=3600,  # Wait up to 1 hour for spot instances
)
```

### 6.3 Auto Scaling

```python
# Configure auto scaling to scale down during low usage
autoscaling.put_scaling_policy(
    PolicyName=f'{endpoint_name}-scale-down',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 30.0,  # Scale down when utilization is low
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

## Step 7: Security Best Practices

### 7.1 IAM Roles and Permissions

```python
# Use least privilege principle
iam_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpoint",
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "*"
        }
    ]
}
```

### 7.2 VPC Configuration

```python
# Deploy in VPC for enhanced security
predictor = estimator.deploy(
    # ... other parameters
    vpc_config={
        'SecurityGroupIds': ['sg-xxxxxxxxx'],
        'Subnets': ['subnet-xxxxxxxxx', 'subnet-yyyyyyyyy']
    }
)
```

## Step 8: Performance Monitoring

### 8.1 CloudWatch Dashboards

```python
# Create CloudWatch dashboard
cloudwatch = boto3.client('cloudwatch')

dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/SageMaker/Endpoints", "Invocations", "EndpointName", endpoint_name]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "us-east-1",
                "title": "Endpoint Invocations"
            }
        }
    ]
}

cloudwatch.put_dashboard(
    DashboardName=f'{endpoint_name}-dashboard',
    DashboardBody=json.dumps(dashboard_body)
)
```

### 8.2 Logging and Debugging

```python
# Enable detailed logging
estimator = TensorFlow(
    # ... other parameters
    debugger_hook_config=False,  # Disable for production
    disable_profiler=True,       # Disable for production
)
```

## Step 9: Cleanup

### 9.1 Delete Endpoint

```python
# Delete endpoint to avoid charges
predictor.delete_endpoint()
print(f"Endpoint {endpoint_name} deleted")
```

### 9.2 Clean Up S3

```python
# Clean up S3 data
s3_client = boto3.client('s3')
s3_client.delete_object(Bucket=bucket, Key='gru-summarizer/data/train_data.npz')
s3_client.delete_object(Bucket=bucket, Key='gru-summarizer/data/val_data.npz')
```

## Best Practices Summary

### Cost Optimization

1. **Use spot instances** for training (up to 90% savings)
2. **Choose appropriate instance types** (GPU for training, CPU for inference)
3. **Enable auto scaling** to scale down during low usage
4. **Delete unused endpoints** to avoid charges

### Performance Optimization

1. **Enable mixed precision training** for faster training
2. **Use optimal batch sizes** for GPU memory utilization
3. **Implement gradient clipping** for stable training
4. **Use learning rate scheduling** for better convergence

### Security

1. **Use least privilege IAM roles**
2. **Deploy in VPC** for enhanced security
3. **Enable encryption** for data at rest and in transit
4. **Regular security updates** for dependencies

### Monitoring

1. **Set up CloudWatch dashboards** for monitoring
2. **Configure auto scaling** based on metrics
3. **Enable detailed logging** for debugging
4. **Set up alerts** for critical metrics

### Scalability

1. **Use auto scaling** for endpoints
2. **Implement load balancing** for high availability
3. **Use multiple availability zones** for redundancy
4. **Monitor resource utilization** and scale accordingly

This setup provides a production-ready, cost-optimized, and scalable solution for running your GRU text summarizer in SageMaker.
