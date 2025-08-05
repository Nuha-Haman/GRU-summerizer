#!/usr/bin/env python3
"""
Upload BBCNews.csv to S3 for SageMaker training
"""

import boto3
import sagemaker
import os

def upload_bbc_data():
    """Upload BBCNews.csv to S3 bucket"""
    
    # Initialize SageMaker session
    session = sagemaker.Session()
    bucket = session.default_bucket()
    
    print(f"📤 Uploading BBCNews.csv to S3...")
    print(f"   - Bucket: {bucket}")
    
    # Check if file exists locally
    if not os.path.exists('BBCNews.csv'):
        print("❌ BBCNews.csv not found in current directory")
        print("Please ensure BBCNews.csv is in the same directory as this script")
        return False
    
    # Upload to S3
    s3_client = boto3.client('s3')
    
    try:
        s3_client.upload_file('BBCNews.csv', bucket, 'BBCNews.csv')
        print(f"✅ Successfully uploaded BBCNews.csv to s3://{bucket}/BBCNews.csv")
        return True
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        return False

if __name__ == "__main__":
    upload_bbc_data() 