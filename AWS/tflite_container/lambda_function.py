import json
import numpy as np
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="sol_pred_mod_NN.tflite")
#interpreter.allocate_tensors()
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()

def lambda_handler(event, context):
    #print("EVENT RECEIVED:", json.dumps(event))
    try:
        # Parse input
        body = json.loads(event['body'])
        #print("BODY:", body)
        features = np.array(body["features"], dtype=np.float32)
        #print("FEATURES SHAPE:", features.shape)

        # adapt shape        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        #print("INPUT DETAILS:", input_details)
        #print("OUTPUT DETAILS:", output_details)

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