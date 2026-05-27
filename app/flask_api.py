from flask import Flask, request, jsonify

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Class names
class_names = [
    'buildings',
    'forest',
    'glacier',
    'mountain',
    'sea',
    'street'
]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load model
model = models.resnet18(pretrained=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 6)

model.load_state_dict(
    torch.load("models/model.pth", map_location=torch.device('cpu'))
)

model.eval()

@app.route('/predict', methods=['POST'])

def predict():

    image = request.files['file']

    image = Image.open(image).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs, 1)

    result = class_names[predicted.item()]

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
