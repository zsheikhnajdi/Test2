import json

def lambda_handler(event, context):
    """
    AWS Lambda function to process user data.
    :param event: Input event data
    :param context: Runtime information
    :return: Processed data
    """
    {
    "data": [1, 2, 3, 4, 5]
}

    data = event['data']
    processed_data = [x * 2 for x in data]
    
    return {
        'statusCode': 200,
        'body': json.dumps(processed_data)
    }
