import json
import os
import boto3
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- S3 Config ---
S3_BUCKET = "solar-prediction"
MODEL_KEY = "models/sol_pred_mod_NN.keras"
TFLITE_KEY = "models/sol_pred_mod_NN.tflite"

s3 = boto3.client("s3")

# --- Utilities ---
def download_model():
    """Download Keras model from S3."""
    local_path = "/tmp/model.keras"
    s3.download_file(S3_BUCKET, MODEL_KEY, local_path)
    model = tf.keras.models.load_model(local_path)
    return model

def upload_model(model, filename):
    """Upload Keras or TFLite model to S3."""
    path = f"/tmp/{os.path.basename(filename)}"
    if filename.endswith(".keras"):
        model.save(path)

    elif filename.endswith(".tflite"):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(path, "wb") as f:
            f.write(tflite_model)
    s3.upload_file(path, S3_BUCKET, filename)

def lambda_handler(event, context):
    try:
        # Load existing model
        model = download_model()

        # Parse incoming JSON body
        body = json.loads(event.get("body", "{}"))
        features = np.array(body.get("features", []), dtype=np.float32)
        targets = np.array(body.get("targets", []), dtype=np.float32)
        
        if len(features) == 0 or len(targets) == 0:
            return {"statusCode": 400, "body": json.dumps({"error": "No features or targets provided"})}

        model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-5))
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True
        )
        model.fit(features, targets, validation_split=0.2, epochs=10, callbacks=[callback])

        # Upload updated Keras and TFLite models
        upload_model(model, MODEL_KEY)
        upload_model(model, TFLITE_KEY)

        return {"statusCode": 200, "body": json.dumps({"status": "success", "updated_rows": len(features)})}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}