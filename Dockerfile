FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system packages required for building/scientific python packages
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   git \
	   curl \
	   ca-certificates \
	   libgomp1 \
	   libopenblas-dev \
	   libsndfile1 \
	   libjpeg-dev \
	   zlib1g-dev \
	&& rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install wheel to avoid build issues
RUN python -m pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
