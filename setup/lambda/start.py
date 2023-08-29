import os
import boto3
import time

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')

    # Get General Configurations
    instance_id = os.environ.get("instance_id")
    script_path = os.environ.get("script_path")
    file_name = os.environ.get("file_name")
    
    # Get Script Parameters
    params = ""
    params+= "--run_name "+os.environ.get("run_name") if os.environ.get("run_name") is not None else ""
    params+= "--data "+os.environ.get("data") if os.environ.get("data") is not None else ""
    params+= "--artefact_dir "+os.environ.get("artefact_dir") if os.environ.get("artefact_dir") is not None else ""
    params+= "--device "+os.environ.get("device") if os.environ.get("device") is not None else ""
    params+= "--use_large " if os.environ.get("use_large") is not None else ""
    params+= "--save_all " if os.environ.get("save_all") is not None else ""
    params+= "--batch_size "+str(os.environ.get("batch_size")) if os.environ.get("batch_size") is not None else ""
    params+= "--stride "+str(os.environ.get("stride")) if os.environ.get("stride") is not None else ""
    params+= "--max_length "+str(os.environ.get("max_length")) if os.environ.get("max_length") is not None else ""
    params+= "--train_fraction "+str(os.environ.get("train_fraction")) if os.environ.get("train_fraction") is not None else ""
    params+= "--epochs "+str(os.environ.get("epochs")) if os.environ.get("epochs") is not None else ""
    params+= "--lr "+str(os.environ.get("lr")) if os.environ.get("lr") is not None else ""
    params+= "--dropout "+str(os.environ.get("dropout")) if os.environ.get("dropout") is not None else ""
    params+= "--clip_grad "+str(os.environ.get("clip_grad")) if os.environ.get("clip_grad") is not None else ""
    params+= "--early_stopping_patience "+str(os.environ.get("early_stopping_patience")) if os.environ.get("early_stopping_patience") is not None else ""
    
    # Start the instance
    ec2.start_instances(InstanceIds=[instance_id])
    
    print("Instance Started")

    while True:
        instance = ec2.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]
        status = instance['State']['Name']
        if status == 'running':
            break
        else:
            time.sleep(5)
    
    print("Starting Training")
    
    ssm = boto3.client("ssm")

    # Run the command on the instance
    ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": ["#!/bin/bash", "source {}/.venv/bin/activate".format(script_path), "python {}/{} {}".format(script_path,file_name,params)]},
    )

###################
# ENV VARS
###################

# {
#     "instance_id": "XXXXXXXXXXXXXXXXXXXXX",
#     "script_path": "/home/ubuntu/receipt-data-extraction/training/",
#     "file_name": "main.py",
#     "run_name": None,
#     "data": None,
#     "artefact_dir": None,
#     "device": None,
#     "use_large": None,
#     "save_all": None,
#     "batch_size": None,
#     "stride": None,
#     "max_length": None,
#     "train_fraction": None,
#     "epochs": None,
#     "lr": None,
#     "dropout": None,
#     "clip_grad": None,
#     "early_stopping_patience": None
# }
