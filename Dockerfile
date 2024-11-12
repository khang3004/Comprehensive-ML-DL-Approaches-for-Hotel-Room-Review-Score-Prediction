# Sử dụng Python 3.9 base image
FROM python:3.9-slim

# Thiết lập working directory
WORKDIR /app

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip và cài đặt các tools cần thiết
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Cài đặt testing dependencies
RUN pip install --no-cache-dir \
    pytest==7.1.1 \
    pytest-cov==2.12.1

# Cài đặt ML dependencies
RUN pip install --no-cache-dir \
    torch==1.9.0 \
    torchvision==0.10.0 \
    numpy==1.21.0 \
    pandas==1.3.0 \
    scikit-learn==0.24.2 \
    statsmodels==0.13.5 \
    scipy==1.7.1 \
    patsy==0.5.2

# Cài đặt visualization dependencies
RUN pip install --no-cache-dir \
    Pillow==8.2.0 \
    matplotlib==3.4.2 \
    seaborn==0.11.1

# Cài đặt web scraping dependencies
RUN pip install --no-cache-dir \
    selenium==4.0.0 \
    beautifulsoup4==4.9.3 \
    requests==2.25.1

# Cài đặt utilities
RUN pip install --no-cache-dir \
    openpyxl==3.1.0 \
    xlrd==2.0.1 \
    python-dotenv==0.19.0 \
    tqdm==4.61.0

# Copy source code
COPY . .

# Tạo cấu trúc thư mục project
RUN mkdir -p \
    data/hotel_images \
    models \
    results \
    logs \
    task_classification \
    task_regression \
    task_clustering \
    task_time_series_forecasting

# Thiết lập environment variables
ENV PYTHONPATH=/app
ENV TASK_DIR=/app/data
ENV MODEL_DIR=/app/models
ENV RESULTS_DIR=/app/results
ENV IMG_DIR=/app/data/hotel_images
ENV PYTHONUNBUFFERED=1

# Default command với đầy đủ arguments
CMD ["python", "-u", "task_regression/evaluate.py", \
    "--task_type", "regression", \
    "--model_type", "ml", \
    "--model", "Ridge_Regression", \
    "--task_dir", "/app/data", \
    "--dataset", "booking_images", \
    "--num_features", "price", "review_count", "Comfort", "Cleanliness", "Facilities", \
    "--cat_features", "star", "Pool", "No_smoking_room", "Families_room", "Room_service", "Free_parking", "Breakfast", "24h_front_desk", "Airport_shuttle", \
    "--img_dir", "/app/data/hotel_images", \
    "--save_model", \
    "--random_state", "42"]
