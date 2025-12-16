# 1. Base image
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements first (for caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy entire project into container
COPY . .

# 6. Expose Flask port
EXPOSE 8000

# 7. Run the Flask app
CMD ["python", "app.py"]
