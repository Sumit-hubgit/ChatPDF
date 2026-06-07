import os
import boto3
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')
response = s3_client.list_buckets()
for bucket in response['Buckets']:
    print(bucket)


response = s3_client.list_objects_v2(Bucket='rag-pipeline1')
objects = response.get('Content',[])
print(objects)
#this is just sample