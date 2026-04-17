from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
from PIL import Image
import io


from src.model import Unet

app = FastAPI()

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet()
model.load_state_dict(torch.load("models/unet.pth", map_location=device))
model.to(device)
model.eval()

def preprocess(image: Image.Image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)


def postprocess(pred):
    pred = torch.argmax(pred, dim=1).squeeze(0)
    return pred.cpu().numpy().astype(np.uint8)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    x = preprocess(image).to(device)

    with torch.no_grad():
        pred = model(x)
    
    mask = postprocess(pred)

    return {
        "mask_shape": mask.shape,
        "unique_classes": np.unique(mask).tolist()
    }