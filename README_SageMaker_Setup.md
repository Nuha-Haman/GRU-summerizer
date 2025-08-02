# GRU Text Summarizer - SageMaker Setup Guide

## 🚀 Quick Start (Minimal Cost)

### Step 1: Install Dependencies

```bash
pip install sagemaker boto3 tensorflow numpy pandas
```

### Step 2: Configure AWS

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, Region, and Output format
```

### Step 3: Run the Script

```bash
python run_sagemaker_gru.py
```

## 💰 Cost-Optimized Configuration

### Training Instance

- **Instance**: `ml.g4dn.xlarge` (1 GPU)
- **Cost**: $0.526/hour (regular), ~$0.15-0.25/hour (spot instances)
- **Savings**: 60-90% with spot instances

### Inference Instance

- **Instance**: `ml.m5.large` (CPU)
- **Cost**: $0.115/hour
- **Alternative**: `ml.c5.large` ($0.085/hour) - even cheaper

### Total Estimated Cost

- **Training**: ~$0.50-1.00 (with spot instances)
- **Inference**: ~$0.115/hour
- **Demo total**: ~$0.60-1.10

## 📊 Instance Comparison

| Instance Type    | Purpose   | Cost/Hour | GPU | vCPU | RAM  |
| ---------------- | --------- | --------- | --- | ---- | ---- |
| `ml.g4dn.xlarge` | Training  | $0.526    | 1   | 4    | 16GB |
| `ml.p3.2xlarge`  | Training  | $3.06     | 1   | 8    | 61GB |
| `ml.m5.large`    | Inference | $0.115    | 0   | 2    | 8GB  |
| `ml.c5.large`    | Inference | $0.085    | 0   | 2    | 4GB  |

## 🎯 How to Run

### Option 1: Run the Complete Script

```bash
# This will do everything automatically
python run_sagemaker_gru.py
```

### Option 2: Step by Step in Python

```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput

# Setup
session = sagemaker.Session()
role = get_execution_role()
bucket = session.default_bucket()

# Create estimator
estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge',  # Cost-optimized
    framework_version='2.10.0',
    py_version='py39',
    hyperparameters={
        'max_len_text': 300,
        'max_len_summary': 30,
        'latent_dim': 256,
        'batch_size': 32,
        'epochs': 5,
        'y_voc_size': 500
    },
    output_path=f's3://{bucket}/gru-summarizer/output',
    use_spot_instances=True,  # Cost savings
    max_wait=3600,
    max_run=3600,
)

# Start training
training_input = TrainingInput(
    s3_data=f's3://{bucket}/gru-summarizer/data',
    content_type='application/x-npy'
)
estimator.fit({'training': training_input})

# Deploy model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',  # Cost-effective inference
    wait=True
)

# Test
result = predictor.predict({'text': 'Test text'})
print(result)

# Cleanup
predictor.delete_endpoint()
```

## 🔧 Customization

### Replace Sample Data

```python
# In run_sagemaker_gru.py, replace create_sample_data() with:
def create_sample_data():
    # Load your actual preprocessed data
    df = pd.read_csv('your_data.csv')

    # Your data preparation logic here
    x_tr = your_preprocessed_input_data
    y_tr = your_preprocessed_output_data
    decoder_input_tr = your_decoder_input_data

    # Save
    np.savez('train_data.npz', x_tr=x_tr, y_tr=y_tr, decoder_input_tr=decoder_input_tr)
    np.savez('val_data.npz', x_val=x_val, y_val=y_val, decoder_input_val=decoder_input_val)
```

### Customize Model Architecture

```python
# In train.py, modify create_gru_model():
def create_gru_model(max_len_text, y_voc_size, latent_dim=256):
    # Your custom GRU architecture here
    # Add attention mechanism, different layers, etc.
    pass
```

### Adjust Hyperparameters

```python
# In the estimator creation:
hyperparameters={
    'max_len_text': 300,      # Your text length
    'max_len_summary': 30,    # Your summary length
    'latent_dim': 256,        # GRU units
    'batch_size': 32,         # Adjust based on GPU memory
    'epochs': 50,             # Training epochs
    'y_voc_size': 5000        # Your vocabulary size
}
```

## 🚨 Important Notes

### Cost Optimization

1. **Use spot instances** for training (saves 60-90%)
2. **Use CPU instances** for inference (cheaper than GPU)
3. **Delete endpoints** when not in use
4. **Set auto-scaling** for production

### Best Practices

1. **Monitor costs** in AWS Console
2. **Use CloudWatch** for monitoring
3. **Set up alerts** for high costs
4. **Clean up resources** regularly

### Troubleshooting

1. **Spot instance failures**: Increase `max_wait` time
2. **Memory issues**: Reduce `batch_size`
3. **Training timeouts**: Increase `max_run` time
4. **Permission errors**: Check IAM roles

## 📈 Production Setup

### Auto Scaling

```python
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
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

### Monitoring

```python
import boto3

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
```

## 🎉 Success!

After running the script, you'll have:

- ✅ Trained GRU model in SageMaker
- ✅ Deployed endpoint for inference
- ✅ Cost-optimized setup
- ✅ Cleanup instructions

**Remember**: Delete the endpoint when not in use to avoid charges!
