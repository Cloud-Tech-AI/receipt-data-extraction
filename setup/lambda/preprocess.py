import boto3

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')

    # Get the instance ID from the event
    instance_id = event['instance_id']

    # Get the script path from the event
    script_path = event['script_path']
    
    # Get the file name from the event
    file_name = event['file_name']

    # Create an SSM client
    ssm = boto3.client("ssm")

    # Run the command on the instance
    ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": ["#!/bin/bash", "source {}/.venv/bin/activate".format(script_path), "python {}/{}".format(script_path,file_name)]},
    )

###########
# INPUT
###########

# {
#   "instance_id": "XXXXXXXXXXXXXXXXXXXXX",
#   "script_path": "/home/ubuntu/receipt-data-extraction/preprocess/",
#   "file_name": "preprocess_textract.py"
# }
