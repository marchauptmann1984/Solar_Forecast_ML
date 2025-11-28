# Start with Amazon Linux 2023 for glibc ≥ 2.34
FROM amazonlinux:2023

# Install Python 3.11 and pip
RUN dnf install -y python3.11 python3.11-pip

# Set Python 3.11 as default
RUN alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    alternatives --install /usr/bin/pip pip /usr/bin/pip3.11 1

# Install Lambda Runtime Interface Client
RUN pip install awslambdaric

# Create working directory
WORKDIR /var/task

# Install dependencies
RUN pip install numpy==1.26.4 tflite-runtime==2.14.0 boto3

# Copy your Lambda function code
COPY lambda_function.py .

# Copy the model
# COPY sol_pred_mod_NN.tflite .

# Set the Lambda handler
CMD ["python", "-m", "awslambdaric", "lambda_function.lambda_handler"]