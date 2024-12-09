# Use the official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
    
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy application files
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Install system dependencies and gcloud SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
       curl apt-transport-https ca-certificates gnupg \
    && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/*


# Add gcloud to PATH
ENV PATH="/usr/lib/google-cloud-sdk/bin:$PATH"

COPY app.py main_app.py welcome_page.py /app/


# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.address=0.0.0.0"]
