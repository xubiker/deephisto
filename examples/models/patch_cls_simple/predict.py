import torch
from models.patch_cls_simple.model import get_model
import sys

if __name__ == "__main__":
    model_weights = "checkpoints/model_weights.pth"
    image_path = sys.argv[1]  # Image file path from command line

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet(num_classes=5)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device).eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = load_image(image_path, mean, std).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(1).item()

    print(f"Predicted Class: {predicted_class}")
