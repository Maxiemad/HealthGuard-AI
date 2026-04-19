# Deployment Guide

## HuggingFace Spaces Deployment

### 1. Prepare Repository

Ensure your repository contains all necessary files:

```
HealthGuard-AI/
|
|--- healthguard_app.py       # Main app (renamed to app.py for HF)
|--- requirements.txt
|--- README.md
|--- .env.example
|--- models/                   # Trained models
|--- rag/indexes/             # Knowledge base indexes
|--- data/                    # Datasets
|--- agent/                   # Agent system
|--- utils/                   # Utilities
```

### 2. HuggingFace Spaces Setup

1. **Create New Space**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" SDK
   - Set Space name (e.g., "healthguard-ai")
   - Choose visibility (Public recommended)

2. **Upload Files**
   ```bash
   git clone https://huggingface.co/spaces/your-username/healthguard-ai
   cd healthguard-ai
   cp -r /path/to/HealthGuard-AI/* .
   git add .
   git commit -m "Initial HealthGuard AI deployment"
   git push
   ```

3. **Set Environment Variables**
   - Go to Space Settings > Repository secrets
   - Add `GROQ_API_KEY` with your Groq API key
   - Space will auto-redeploy

### 3. File Modifications for HF Spaces

#### Rename app file:
```bash
mv healthguard_app.py app.py
```

#### Update requirements.txt for HF:
```
streamlit==1.32.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
plotly==5.18.0
shap==0.44.0
groq==0.11.0
langgraph==0.0.26
sentence-transformers==2.2.2
faiss-cpu==1.7.4
reportlab==4.0.7
xgboost==2.0.3
networkx==3.1
scipy==1.11.1
```

#### Add HF Spaces metadata to README.md:
```yaml
---
title: HealthGuard AI
emoji:  1f7e5
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---
```

### 4. Troubleshooting

#### Common Issues:
1. **Model loading errors**: Ensure all model files are in `models/` directory
2. **Memory issues**: HF Spaces has limited RAM, consider model optimization
3. **API key errors**: Set `GROQ_API_KEY` in Repository secrets
4. **Import errors**: Check all dependencies in requirements.txt

#### Debug Mode:
Add to app.py for debugging:
```python
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
```

### 5. Performance Optimization

#### Model Caching:
```python
@st.cache_resource
def load_models():
    # Model loading code
    return models
```

#### Lazy Loading:
Load heavy models only when needed to reduce startup time.

### 6. Custom Domain (Optional)

1. Go to Space Settings
2. Add custom domain
3. Configure DNS records

### 7. Monitoring

- Check Space logs for errors
- Monitor usage metrics in HF dashboard
- Set up alerts for high error rates

## Local Production Deployment

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t healthguard-ai .
docker run -p 8501:8501 healthguard-ai
```

### Kubernetes Deployment

Create `k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: healthguard-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: healthguard-ai
  template:
    metadata:
      labels:
        app: healthguard-ai
    spec:
      containers:
      - name: healthguard-ai
        image: healthguard-ai:latest
        ports:
        - containerPort: 8501
        env:
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: groq-api-key
---
apiVersion: v1
kind: Service
metadata:
  name: healthguard-ai-service
spec:
  selector:
    app: healthguard-ai
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

## Security Considerations

1. **API Keys**: Never commit API keys to git
2. **Patient Data**: Implement data anonymization
3. **Access Control**: Add authentication if needed
4. **HTTPS**: Ensure SSL/TLS encryption
5. **Input Validation**: Sanitize all user inputs

## Scaling

### Horizontal Scaling
- Load balancer with multiple app instances
- Shared model storage (S3, GCS)
- Database for user sessions

### Vertical Scaling
- Increase memory for larger models
- GPU acceleration for model inference
- Faster storage (SSD)

## Backup and Recovery

1. **Model Artifacts**: Regular backups to cloud storage
2. **Knowledge Base**: Version control for documents
3. **User Data**: Database backups
4. **Configuration**: Git repository

## Maintenance

1. **Model Updates**: Retrain models with new data
2. **Dependency Updates**: Regular package updates
3. **Security Patches**: Keep dependencies secure
4. **Performance Monitoring**: Track response times and errors
