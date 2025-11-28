# -------------------------------
# Stage 1: Build dependencies
# -------------------------------
FROM amazonlinux:2023 AS builder

# Install Python 3.11 and system deps for TF
RUN dnf install -y python3.11 python3.11-pip gcc gcc-c++ make wget tar unzip \
                   libjpeg-turbo-devel zlib-devel && \
    alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    alternatives --install /usr/bin/pip pip /usr/bin/pip3.11 1 && \
    dnf clean all

# Install Python packages into /install
RUN pip install --prefix=/install --no-cache-dir tensorflow-cpu==2.18.1 numpy==1.26.4 boto3 awslambdaric

# -------------------------------
# Stage 2: Final image
# -------------------------------
FROM amazonlinux:2023

# Set up Python 3.11
RUN dnf install -y python3.11 python3.11-pip && \
    alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    alternatives --install /usr/bin/pip pip /usr/bin/pip3.11 1 && \
    dnf clean all

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /var/task

# Copy Lambda function code
COPY lambda_function.py .

# Set Lambda handler
CMD ["python", "-m", "awslambdaric", "lambda_function.lambda_handler"]