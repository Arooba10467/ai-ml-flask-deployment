# AI ML Deployment Project (Flask + Docker)

## Project Overview
This project demonstrates the ML lifecycle including training, inference, and deployment using Flask API and Docker.

---

## Dataset
- Dataset contains 6 classes: buildings, forest, glacier, mountain, sea, street
- Used for image classification using pretrained ResNet model

---

## Model
- Pretrained ResNet (PyTorch)
- Fine-tuned on custom dataset
- Accuracy: ~85%

---

## Project Structure
- train.py → model training
- inference.py → batch prediction
- flask_api.py → REST API
- Dockerfile → containerization

---

## API Usage

### Endpoint:
POST /predict

### Input:
form-data:
- file: image

### Output:
```json
{
  "prediction": "forest"
}
