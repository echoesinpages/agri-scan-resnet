# Use a lightweight Python 3.9 image to keep the container small
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependency file first (layer caching — dependencies rarely change)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Copy the trained model (must be in project root before building)
COPY model.h5 .

# Expose Streamlit's default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
