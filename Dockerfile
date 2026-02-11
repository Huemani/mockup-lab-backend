FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
```

4. Klikk **"Commit changes"**

---

## Steg 2: Deploy til Railway

### 2.1 Opprett Railway Account
1. Gå til https://railway.app
2. Klikk **"Start a New Project"**
3. Logg inn med GitHub

### 2.2 Deploy Backend
1. Velg **"Deploy from GitHub repo"**
2. Velg `mockup-lab-backend` repository
3. Railway vil automatisk detektere Dockerfile

### 2.3 Sett Environment Variables
1. I Railway dashboard, gå til ditt prosjekt
2. Klikk på **"Variables"** tab
3. Legg til disse variablene:
```
CLOUDINARY_CLOUD_NAME=ducsuev69
CLOUDINARY_API_KEY=<din_api_key_fra_cloudinary>
CLOUDINARY_API_SECRET=<din_api_secret_fra_cloudinary>
PORT=8000
