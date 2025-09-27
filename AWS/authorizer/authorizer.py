import boto3
#import json
dynamodb = boto3.client('dynamodb')
TABLE_NAME = "ApiTokens"

def lambda_handler(event, context):
    token = event['headers'].get('x-api-key')
    #print("EVENT RECEIVED:", json.dumps(event))
    if not token:
        return {"isAuthorized": False}

    try:
        response = dynamodb.get_item(
            TableName=TABLE_NAME,
            Key={"token": {"S": token}}
        )
        if "Item" in response:
            return {"isAuthorized": True}
    except Exception as e:
        print("Error checking token:", e)

    return {"isAuthorized": False}