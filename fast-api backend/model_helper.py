import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights="DEFAULT")

        for params in self.model.parameters():
            params.requires_grad = False
        
        for params in self.model.layer4.parameters():
            params.requires_grad = True
        
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential( # type: ignore
            nn.Dropout(0.5),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.model(x)

tranformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],        
        std=[0.229, 0.224, 0.225]          
    )
])
classes = ["Normal", "Pneumonia"]

def give_prediction(image_pth):
    image = Image.open(image_pth).convert("RGB")
    image = tranformations(image).unsqueeze(0).to(device) # type: ignore

    model = Model().to(device)
    model.load_state_dict(torch.load("../artifacts/model_2.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        predictions = model(image)                # logits
        probs = torch.softmax(predictions, dim=1) # convert to probabilities
        confidence, predicted_label = torch.max(probs, 1)

    return {
        "class": classes[predicted_label.item()], # type: ignore
        "probability": confidence.item()
    }
