FROM public.ecr.aws/lambda/python:3.12

# Copy requirements first to leverage Docker cache
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the lambdas directory
RUN mkdir -p ${LAMBDA_TASK_ROOT}/lambdas

# IMPORTANT: Copy Python files to lambdas directory
COPY lambdas/ ${LAMBDA_TASK_ROOT}/lambdas/

# Copy supporting directories
COPY utils/ ${LAMBDA_TASK_ROOT}/utils/
COPY retrieval/ ${LAMBDA_TASK_ROOT}/retrieval/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

# Create __init__.py files to make directories into proper Python packages
RUN touch ${LAMBDA_TASK_ROOT}/lambdas/__init__.py
RUN touch ${LAMBDA_TASK_ROOT}/utils/__init__.py
RUN touch ${LAMBDA_TASK_ROOT}/retrieval/__init__.py

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# No need for a default command - Lambda will use the one defined in the function