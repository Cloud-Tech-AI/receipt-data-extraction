import boto3

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')

    # Get the instance ID from the event
    instance_id = event["instance_id"]

    # Start the instance
    ec2.stop_instances(InstanceIds=[instance_id])

    print('Instance stopped')

###########
# INPUT
###########

# {
#     "instance_id": "XXXXXXXXXXXXXXXXXXXXX"
# }