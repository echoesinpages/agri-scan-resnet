# Use a lightweight Python 3.9 image to keep the container small
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependency file first (to cache layers and speed up builds)
COPY requirements.txt .

# Install dependencies
# We add --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ src/

# Copy the AI Model (Critical: The model must be in your project root folder)
COPY model.h5 .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the app when the container starts
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
