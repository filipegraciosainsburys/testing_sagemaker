
import boto3
import re
from sagemaker import get_execution_role
import io
import numpy as np
import sagemaker.amazon.common as smac
import os
import sagemaker



# get aws credentials
with open('aws_credentials') as f:
    lines = f.readlines()
    user_access_key = lines[1].split('=')[1][:-1].strip()
    user_secret_access_key = lines[2].split('=')[1][:-1].strip()

print('user key', user_access_key)
print('secret user key', user_secret_access_key)



executor_role =
krang_role =
bucket =
prefix =


import pickle, gzip, numpy, urllib.request, json


# Load the dataset
exists = os.path.isfile("mnist.pkl")
if not exists:
    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
with open('mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


vectors = np.array([t.tolist() for t in train_set[0]]).astype('float32')
labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1, 0).astype('float32')

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)


# link to aws account with my credentials
sts_client = boto3.client('sts',
                          aws_access_key_id=user_access_key,
                          aws_secret_access_key=user_secret_access_key,
                          region_name='eu-west-1')

# assume the krang dev role
response = sts_client.assume_role(
    RoleArn=krang_role,
    RoleSessionName='blablabla')


# create a boto sessions
access_key = response['Credentials']['AccessKeyId']
secret_access_key = response['Credentials']['SecretAccessKey']
session_token = response['Credentials']['SessionToken']

boto_session = boto3.Session(
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_access_key,
    aws_session_token=session_token,
    region_name='eu-west-1')



key = 'recordio-pb-data'
upload_path = os.path.join(prefix, 'train', key)
boto_session.resource('s3').Bucket(bucket).Object(upload_path).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))

#session.resource('s3').Bucket(bucket).download_file('sagemaker/DEMO-ntm-synthetic/output/ntm-2018-12-27-10-56-01-889/output/model.tar.gz', 'model.tar.gz_')
output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))

from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'linear-learner')
print(container)


sess = sagemaker.Session(boto_session=boto_session)

linear = sagemaker.estimator.Estimator(container,
                                       executor_role,
                                       train_instance_count=1,
                                       train_instance_type='ml.c4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess)

linear.set_hyperparameters(feature_dim=784,
                           predictor_type='binary_classifier',
                           mini_batch_size=200)

linear.fit({'train': s3_train_data})