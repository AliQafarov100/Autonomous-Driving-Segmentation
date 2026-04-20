from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import torch
import numpy as np
from PIL import Image
import io


from src.model import UNet

app = FastAPI()

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet()
model.load_state_dict(torch.load("models/unet.pth", map_location=device))
model.to(device)
model.eval()

def colorize(mask):
    COLORS = {
        0: (0, 0, 0),
        1: (128, 0, 0),
        2: (0, 128, 0),
        3: (128, 128, 0),
        4: (0, 0, 128),
        5: (128, 0, 128),
        6: (0, 128, 128),
        7: (128, 128, 128),
        8: (64, 0, 0),
        9: (192, 0, 0),
        10: (64, 128, 0),
        11: (192, 128, 0),
        12: (64, 0, 128),
    }

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in COLORS.items():
        color_mask[mask == cls] = color

    return color_mask

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
    mask = colorize(mask)

    mask_img = Image.fromarray(mask)

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")