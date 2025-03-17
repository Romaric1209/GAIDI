# Use a lightweight Python image
FROM python:3.10.6-buster

# Set the working directory in the container
WORKDIR /app

# Copy project files to the container
COPY . /app  # This copies everything from the current directory to /app in the container

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

# Expose the FastAPI default port
EXPOSE 8000

# Use JSON syntax for CMD to avoid OS signal issues
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
