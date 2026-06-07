import os
import boto3
from .config import Config
# s3_client = boto3.client('s3')
# s3_resource = boto3.resource('s3')
# response = s3_client.list_buckets()
# for bucket in response['Buckets']:
#     print(bucket)


# response = s3_client.list_objects_v2(Bucket='rag-pipeline1')
# objects = response.get('Content',[])
# print(objects)
#this is just sample


s3_client = boto3.client('s3')
response = s3_client.list_buckets()
#print(response)
for bucket in response['Buckets']:
    print(bucket['Name'])

class AwsOperations:
    def __init__(self,config:Config):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id = config.aws_access_key,
            aws_secret_access_key = config.aws_secret_key,
            region_name = config.aws_region
        )
    def upload_file(self,filename:str,file_obj,folder:str=""):
        key = f"{folder}/{filename}" if folder else filename
        self.s3_client.upload_fileobj(
            file_obj,
            self.config.aws_bucket,
            key
        )
        return {
            "message": "Upload successful",
            "s3_key": key
        }