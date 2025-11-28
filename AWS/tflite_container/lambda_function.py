import json
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import boto3
import os
import tempfile

# --- Configuration ---
S3_BUCKET = "solar-prediction"
S3_KEY = "models/sol_pred_mod_NN.tflite"
LOCAL_MODEL_PATH = "/tmp/sol_pred_mod_NN.tflite"
LOCAL_ETAG_PATH = "/tmp/model_etag.txt"
LAST_CHECK_PATH = "/tmp/last_check.txt"
CHECK_INTERVAL = 600  # 10 minutes


s3 = boto3.client("s3")

# --- Utilities ---
def should_check_s3() -> bool:
    """Avoid checking S3 too often to save cost/latency."""
    if not os.path.exists(LAST_CHECK_PATH):
        return True
    with open(LAST_CHECK_PATH) as f:
        last = float(f.read().strip())
    return (time.time() - last) > CHECK_INTERVAL

def update_last_check():
    with open(LAST_CHECK_PATH, "w") as f:
        f.write(str(time.time()))

def model_updated_in_s3() -> bool:
    """Compare local ETag with S3 to detect model updates."""
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
        new_etag = head['ETag'].strip('"')
    except Exception as e:
        print("Error checking S3:", e)
        return False

    old_etag = None
    if os.path.exists(LOCAL_ETAG_PATH):
        with open(LOCAL_ETAG_PATH) as f:
            old_etag = f.read().strip()

    if new_etag != old_etag:
        print(f"Model update detected (old={old_etag}, new={new_etag})")
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)
        with open(LOCAL_ETAG_PATH, "w") as f:
            f.write(new_etag)
        return True
    else:
        print("Model unchanged")
        return False

def get_interpreter():
    """Load the TFLite model into memory."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("No local model — downloading from S3...")
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)
    interpreter = tflite.Interpreter(model_path=LOCAL_MODEL_PATH)
    return interpreter

# --- Cold start: initial model load ---
print("Initializing Lambda with TFLite model...")
interpreter = get_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model ready.")

def lambda_handler(event, context):
    global interpreter, input_details, output_details
    #print("EVENT RECEIVED:", json.dumps(event))
    try:
        # Parse input
        if should_check_s3():
            update_last_check()
            if model_updated_in_s3():
                interpreter = get_interpreter()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print("Model reloaded successfully.")
        
        body = json.loads(event['body'])
        #print("BODY:", body)
        features = np.array(body["features"], dtype=np.float32)
        #print("FEATURES SHAPE:", features.shape)

        interpreter.resize_tensor_input(input_details[0]['index'], features.shape)
        interpreter.allocate_tensors()

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index']).tolist()
        #print("PREDICTIONS:", predictions)

        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': predictions})
        }

    except Exception as e:
        print("ERROR:", repr(e))
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }