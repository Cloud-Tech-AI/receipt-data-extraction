import json
from preprocess_textract import PreProcessTextract

def lambda_handler(event, context):
    preprocess = PreProcessTextract()
    preprocess.process()
    print('TESTING CHANGES')
    return {
        'statusCode': 200,
        'body': json.dumps('Data Processed Successfully!')
    }
