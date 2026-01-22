# 1. Base Image
FROM python:3.9-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Dependency Manifest
COPY requirements.txt .

# 5. Install Python Libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Expose Port
EXPOSE 5000

# 8. Entrypoint
CMD ["python", "src/app.py"]
