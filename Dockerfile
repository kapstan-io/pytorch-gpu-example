# Use a slim Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files
COPY main.py requirements.txt ./

# Install PyTorch and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run simplenet.py when the container launches
CMD ["python", "main.py"]