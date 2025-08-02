"""
SageMaker Setup for GRU-based Text Summarizer
Best Practices Implementation
"""

import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput
import boto3
import json
from datetime import datetime

class SageMakerGRUSummarizer:
    def __init__(self, region='us-east-1'):
        """
        Initialize SageMaker session and configuration
        """
        self.region = region
        self.sagemaker_session = sagemaker.Session()
        self.role = get_execution_role()
        self.bucket = self.sagemaker_session.default_bucket()
        
        print(f"SageMaker Session initialized:")
        print(f"- Region: {self.region}")
        print(f"- Role: {self.role}")
        print(f"- Bucket: {self.bucket}")
    
    def create_training_script(self):
        """
        Create the training script for SageMaker
        """
        training_script = '''
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    Input, GRU, Embedding, Dense, Concatenate, Dropout,
    TimeDistributed, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
import sagemaker
from sagemaker import get_execution_role

# Enable mixed precision for better performance
mixed_precision.set_global_policy('mixed_float16')

# Memory optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def load_data(data_dir):
    """Load training data from SageMaker data channels"""
    train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    val_data = np.load(os.path.join(data_dir, 'val_data.npz'))
    
    return (train_data['x_tr'], train_data['y_tr'], train_data['decoder_input_tr']), \
           (val_data['x_val'], val_data['y_val'], val_data['decoder_input_val'])

def create_optimized_gru_model(max_len_text, y_voc_size, latent_dim=256, embedding_dim=300):
    """Create optimized GRU model"""
    K.clear_session()
    
    # Encoder
    encoder_text_input = Input(shape=(max_len_text,), name='encoder_text_input')
    enc_emb = Embedding(
        input_dim=encoder_embedding_matrix.shape[0],
        output_dim=encoder_embedding_matrix.shape[1],
        weights=[encoder_embedding_matrix],
        trainable=False,
        name="encoder_embedding"
    )(encoder_text_input)

    # GRU layers with optimizations
    encoder_gru1 = GRU(
        latent_dim, 
        return_sequences=True, 
        return_state=True, 
        name="encoder_gru1",
        dropout=0.3,
        recurrent_dropout=0.3
    )
    encoder_output1, _ = encoder_gru1(enc_emb)
    encoder_output1 = BatchNormalization()(encoder_output1)

    encoder_gru2 = GRU(
        latent_dim, 
        return_sequences=True, 
        return_state=True, 
        name="encoder_gru2",
        dropout=0.3,
        recurrent_dropout=0.3
    )
    encoder_output2, _ = encoder_gru2(encoder_output1)
    encoder_output2 = BatchNormalization()(encoder_output2)

    encoder_gru3 = GRU(
        latent_dim, 
        return_sequences=True, 
        return_state=True, 
        name="encoder_gru3",
        dropout=0.3,
        recurrent_dropout=0.3
    )
    encoder_outputs, state_h = encoder_gru3(encoder_output2)
    encoder_outputs = BatchNormalization()(encoder_outputs)

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    dec_emb = Embedding(
        input_dim=decoder_embedding_matrix.shape[0],
        output_dim=decoder_embedding_matrix.shape[1],
        weights=[decoder_embedding_matrix],
        trainable=True,
        name="decoder_embedding"
    )(decoder_inputs)

    decoder_gru = GRU(
        latent_dim, 
        return_sequences=True, 
        return_state=True, 
        name="decoder_gru",
        dropout=0.3,
        recurrent_dropout=0.3
    )
    decoder_outputs, _ = decoder_gru(dec_emb, initial_state=state_h)
    decoder_outputs = BatchNormalization()(decoder_outputs)

    # Attention mechanism
    attn_layer = OptimizedAttentionLayer(name='attention_layer')
    attn_out, _ = attn_layer([encoder_outputs, decoder_outputs])

    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])
    decoder_concat_input = Dropout(0.3)(decoder_concat_input)
    
    decoder_dense = TimeDistributed(
        Dense(y_voc_size, activation='softmax'), 
        name="final_dense"
    )
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model(inputs=[encoder_text_input, decoder_inputs], outputs=decoder_outputs)
    return model

class OptimizedAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OptimizedAttentionLayer, self).__init__(**kwargs)

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
        super(OptimizedAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        encoder_outputs, decoder_outputs = inputs
        decoder_expanded = tf.expand_dims(decoder_outputs, 2)
        encoder_expanded = tf.expand_dims(encoder_outputs, 1)
        
        score = K.tanh(
            tf.linalg.matmul(encoder_expanded, self.W_a) + 
            tf.linalg.matmul(decoder_expanded, self.U_a)
        )
        
        score = tf.linalg.matmul(score, self.V_a)
        score = tf.squeeze(score, axis=-1)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(attention_weights, encoder_outputs)
        
        return context, attention_weights

def train_model():
    """Main training function for SageMaker"""
    # Load hyperparameters
    hyperparameters = json.loads(os.environ.get('SM_HYPERPARAMETERS', '{}'))
    
    max_len_text = hyperparameters.get('max_len_text', 300)
    max_len_summary = hyperparameters.get('max_len_summary', 30)
    latent_dim = hyperparameters.get('latent_dim', 256)
    batch_size = hyperparameters.get('batch_size', 64)
    epochs = hyperparameters.get('epochs', 50)
    learning_rate = hyperparameters.get('learning_rate', 1e-3)
    
    # Load data
    train_data, val_data = load_data(os.environ['SM_CHANNEL_TRAINING'])
    
    # Create model
    model = create_optimized_gru_model(max_len_text, y_voc_size, latent_dim)
    
    # Optimized optimizer
    optimizer = AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-5,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(os.environ['SM_MODEL_DIR'], 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        {'encoder_text_input': train_data[0], 'decoder_input': train_data[2]},
        train_data[1],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            {'encoder_text_input': val_data[0], 'decoder_input': val_data[2]},
            val_data[1]
        ),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save(os.path.join(os.environ['SM_MODEL_DIR'], 'final_model.keras'))
    
    # Save training history
    with open(os.path.join(os.environ['SM_MODEL_DIR'], 'training_history.json'), 'w') as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    train_model()
'''
        
        with open('train.py', 'w') as f:
            f.write(training_script)
        
        print("Training script created: train.py")
    
    def create_requirements_txt(self):
        """
        Create requirements.txt for SageMaker
        """
        requirements = '''
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
gensim>=4.0.0
tqdm>=4.62.0
rouge-score>=0.1.2
nltk>=3.6.0
bert-score>=0.3.11
'''
        
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        
        print("Requirements file created: requirements.txt")
    
    def create_dockerfile(self):
        """
        Create Dockerfile for custom container (optional)
        """
        dockerfile = '''
FROM tensorflow/tensorflow:2.10.0-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml/code

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy training script
COPY train.py .

# Set environment variables
ENV PYTHONPATH=/opt/ml/code
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Default command
CMD ["python", "train.py"]
'''
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile)
        
        print("Dockerfile created: Dockerfile")
    
    def create_inference_script(self):
        """
        Create inference script for SageMaker endpoint
        """
        inference_script = '''
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def model_fn(model_dir):
    """Load the trained model"""
    model = load_model(os.path.join(model_dir, 'best_model.keras'))
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        text = input_data['text']
        return text
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Generate summary prediction"""
    # Tokenize input text
    tokenized_text = tokenizer.texts_to_sequences([input_data])
    padded_text = pad_sequences(tokenized_text, maxlen=max_len_text, padding='post')
    
    # Generate summary
    summary = generate_summary(padded_text, model)
    
    return summary

def output_fn(prediction, accept):
    """Format output"""
    if accept == 'application/json':
        return json.dumps({'summary': prediction}), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

def generate_summary(input_sequence, model, max_len_summary=30):
    """Generate summary using the trained model"""
    # This is a simplified version - you'll need to implement the full inference logic
    # based on your encoder-decoder architecture
    
    # For now, return a placeholder
    return "Generated summary placeholder"
'''
        
        with open('inference.py', 'w') as f:
            f.write(inference_script)
        
        print("Inference script created: inference.py")
    
    def setup_training_job(self, instance_type='ml.p3.2xlarge', instance_count=1):
        """
        Setup SageMaker training job with best practices
        """
        # Create TensorFlow estimator
        estimator = TensorFlow(
            entry_point='train.py',
            role=self.role,
            instance_count=instance_count,
            instance_type=instance_type,
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
            output_path=f's3://{self.bucket}/gru-summarizer/output',
            code_location=f's3://{self.bucket}/gru-summarizer/code',
            use_spot_instances=True,  # Use spot instances for cost optimization
            max_wait=3600,  # Maximum wait time for spot instances
            max_run=7200,   # Maximum training time (2 hours)
            debugger_hook_config=False,  # Disable debugger for faster training
            disable_profiler=True,  # Disable profiler for faster training
            keep_alive_period_in_seconds=1800,  # Keep instance alive for 30 minutes
        )
        
        return estimator
    
    def create_training_data(self, data_path):
        """
        Upload training data to S3
        """
        # This is a placeholder - you'll need to implement data preparation
        # and upload to S3 based on your specific data format
        
        training_input = TrainingInput(
            s3_data=f's3://{self.bucket}/gru-summarizer/data',
            content_type='application/x-npy'
        )
        
        return training_input
    
    def deploy_model(self, model, endpoint_name=None):
        """
        Deploy model to SageMaker endpoint with best practices
        """
        if endpoint_name is None:
            endpoint_name = f'gru-summarizer-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        
        # Create predictor with optimized configuration
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large',  # CPU instance for inference
            endpoint_name=endpoint_name,
            accelerator_type=None,  # No GPU for inference to save costs
            data_capture_config=None,  # Disable data capture for now
            wait=True
        )
        
        return predictor
    
    def create_monitoring_setup(self, endpoint_name):
        """
        Setup CloudWatch monitoring for the endpoint
        """
        cloudwatch = boto3.client('cloudwatch', region_name=self.region)
        
        # Create custom metrics for monitoring
        metrics = [
            {
                'MetricName': 'Invocations',
                'Namespace': f'Custom/{endpoint_name}',
                'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}]
            },
            {
                'MetricName': 'ModelLatency',
                'Namespace': f'Custom/{endpoint_name}',
                'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}]
            }
        ]
        
        print(f"CloudWatch monitoring setup for endpoint: {endpoint_name}")
        return metrics

def main():
    """
    Main function to setup SageMaker training and deployment
    """
    print("Setting up SageMaker GRU Text Summarizer...")
    
    # Initialize SageMaker setup
    sm_setup = SageMakerGRUSummarizer()
    
    # Create necessary files
    sm_setup.create_training_script()
    sm_setup.create_requirements_txt()
    sm_setup.create_dockerfile()
    sm_setup.create_inference_script()
    
    # Setup training job
    estimator = sm_setup.setup_training_job()
    
    # Setup training data (you'll need to implement this based on your data)
    training_input = sm_setup.create_training_data('your_data_path')
    
    print("\\nSageMaker setup completed!")
    print("\\nNext steps:")
    print("1. Prepare your training data and upload to S3")
    print("2. Run the training job: estimator.fit({'training': training_input})")
    print("3. Deploy the model: predictor = sm_setup.deploy_model(estimator)")
    print("4. Test the endpoint with sample data")
    
    return sm_setup, estimator

if __name__ == "__main__":
    main()
'''
        
        with open('sagemaker_setup.py', 'w') as f:
            f.write(sagemaker_setup)
        
        print("SageMaker setup script created: sagemaker_setup.py")
    
    def create_training_notebook(self):
        """
        Create a Jupyter notebook for SageMaker training
        """
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# SageMaker GRU Text Summarizer Training\n",
                        "## Best Practices Implementation"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import sagemaker\n",
                        "from sagemaker import get_execution_role\n",
                        "from sagemaker.tensorflow import TensorFlow\n",
                        "from sagemaker.inputs import TrainingInput\n",
                        "import boto3\n",
                        "import json\n",
                        "from datetime import datetime\n",
                        "\n",
                        "# Initialize SageMaker session\n",
                        "sagemaker_session = sagemaker.Session()\n",
                        "role = get_execution_role()\n",
                        "bucket = sagemaker_session.default_bucket()\n",
                        "\n",
                        "print(f\"SageMaker Session: {sagemaker_session}\")\n",
                        "print(f\"Role: {role}\")\n",
                        "print(f\"Bucket: {bucket}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Setup training estimator with best practices\n",
                        "estimator = TensorFlow(\n",
                        "    entry_point='train.py',\n",
                        "    role=role,\n",
                        "    instance_count=1,\n",
                        "    instance_type='ml.p3.2xlarge',  # GPU instance for training\n",
                        "    framework_version='2.10.0',\n",
                        "    py_version='py39',\n",
                        "    hyperparameters={\n",
                        "        'max_len_text': 300,\n",
                        "        'max_len_summary': 30,\n",
                        "        'latent_dim': 256,\n",
                        "        'batch_size': 64,\n",
                        "        'epochs': 50,\n",
                        "        'learning_rate': 1e-3\n",
                        "    },\n",
                        "    output_path=f's3://{bucket}/gru-summarizer/output',\n",
                        "    code_location=f's3://{bucket}/gru-summarizer/code',\n",
                        "    use_spot_instances=True,  # Cost optimization\n",
                        "    max_wait=3600,\n",
                        "    max_run=7200,\n",
                        "    debugger_hook_config=False,\n",
                        "    disable_profiler=True\n",
                        ")\n",
                        "\n",
                        "print(\"Training estimator created successfully!\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Prepare training data input\n",
                        "training_input = TrainingInput(\n",
                        "    s3_data=f's3://{bucket}/gru-summarizer/data',\n",
                        "    content_type='application/x-npy'\n",
                        ")\n",
                        "\n",
                        "# Start training job\n",
                        "print(\"Starting training job...\")\n",
                        "estimator.fit({'training': training_input})\n",
                        "print(\"Training completed!\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Deploy model to endpoint\n",
                        "endpoint_name = f'gru-summarizer-{datetime.now().strftime(\"%Y%m%d-%H%M%S\")}'\n",
                        "\n",
                        "predictor = estimator.deploy(\n",
                        "    initial_instance_count=1,\n",
                        "    instance_type='ml.m5.large',  # CPU for inference\n",
                        "    endpoint_name=endpoint_name,\n",
                        "    wait=True\n",
                        ")\n",
                        "\n",
                        "print(f\"Model deployed to endpoint: {endpoint_name}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Test the endpoint\n",
                        "test_data = {\n",
                        "    'text': 'This is a sample text for testing the GRU summarizer.'\n",
                        "}\n",
                        "\n",
                        "try:\n",
                        "    result = predictor.predict(test_data)\n",
                        "    print(f\"Prediction: {result}\")\n",
                        "except Exception as e:\n",
                        "    print(f\"Error testing endpoint: {e}\")\n",
                        "\n",
                        "# Clean up (optional)\n",
                        "# predictor.delete_endpoint()"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        import json
        with open('sagemaker_training.ipynb', 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        print("Training notebook created: sagemaker_training.ipynb")

def main():
    """
    Main function to setup SageMaker training and deployment
    """
    print("Setting up SageMaker GRU Text Summarizer...")
    
    # Initialize SageMaker setup
    sm_setup = SageMakerGRUSummarizer()
    
    # Create necessary files
    sm_setup.create_training_script()
    sm_setup.create_requirements_txt()
    sm_setup.create_dockerfile()
    sm_setup.create_inference_script()
    sm_setup.create_training_notebook()
    
    # Setup training job
    estimator = sm_setup.setup_training_job()
    
    # Setup training data (you'll need to implement this based on your data)
    training_input = sm_setup.create_training_data('your_data_path')
    
    print("\nSageMaker setup completed!")
    print("\nNext steps:")
    print("1. Prepare your training data and upload to S3")
    print("2. Run the training job: estimator.fit({'training': training_input})")
    print("3. Deploy the model: predictor = sm_setup.deploy_model(estimator)")
    print("4. Test the endpoint with sample data")
    
    return sm_setup, estimator

if __name__ == "__main__":
    main() 