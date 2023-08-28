import boto3
import time

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')

    # Get General Configurations
    instance_id = event["instance_id"]
    script_path = event["script_path"]
    file_name = event["file_name"]
    
    # Get Script Parameters
    params = ""
    params+= "--run_name "+event["run_name"] if event["run_name"] is not None else ""
    params+= "--data "+event["data"] if event["data"] is not None else ""
    params+= "--artefact_dir "+event["artefact_dir"] if event["artefact_dir"] is not None else ""
    params+= "--device "+event["device"] if event["device"] is not None else ""
    params+= "--use_large " if event["use_large"] is not None else ""
    params+= "--save_all " if event["save_all"] is not None else ""
    params+= "--batch_size "+str(event["batch_size"]) if event["batch_size"] is not None else ""
    params+= "--stride "+str(event["stride"]) if event["stride"] is not None else ""
    params+= "--max_length "+str(event["max_length"]) if event["max_length"] is not None else ""
    params+= "--train_fraction "+str(event["train_fraction"]) if event["train_fraction"] is not None else ""
    params+= "--epochs "+str(event["epochs"]) if event["epochs"] is not None else ""
    params+= "--lr "+str(event["lr"]) if event["lr"] is not None else ""
    params+= "--dropout "+str(event["dropout"]) if event["dropout"] is not None else ""
    params+= "--clip_grad "+str(event["clip_grad"]) if event["clip_grad"] is not None else ""
    params+= "--early_stopping_patience "+str(event["early_stopping_patience"]) if event["early_stopping_patience"] is not None else ""
    
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
