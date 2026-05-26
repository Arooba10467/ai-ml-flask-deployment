import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Class names
class_names = [
    'buildings',
    'forest',
    'glacier',
    'mountain',
    'sea',
    'street'
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load model
model = models.resnet18(pretrained=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 6)

model.load_state_dict(
    torch.load("model.pth", map_location=torch.device('cpu'))
)

model.eval()

# Prediction function
def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

# Example usage
if __name__ == "__main__":

    prediction = predict_image("sample.jpg")

    print("Prediction:", prediction)