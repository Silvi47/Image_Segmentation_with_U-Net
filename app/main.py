import torch
import base64
from fastapi import FastAPI, UploadFile, File, Body
from skimage import io as skimage_io, transform
import io
from typing import List
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2
from config import *
from model import UnetMod
from preprocessing import *

app = FastAPI()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

model = UnetMod().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model1, optimizer1, start_epoch, valid_loss_min = load_ckp(CHECKPOINT_PATH, model, optimizer)

transforms = Compose([Normalize(mean, std), ToTensorV2()])

@app.post("/")
async def nucleiSegmentation(files: List[UploadFile] = File(...)):
    masks = []
    for img in files:
        f = img.file.read()
        f = io.BytesIO(f)
        
        image = skimage_io.imread(f)[:, :, :3].astype('float32')
        image = transform.resize(image, (128, 128))
        img_normalized = transforms(image=image)['image']
        img_normalized = img_normalized.unsqueeze(0)
        
        with torch.no_grad():
            y_pred = model1.forward(img_normalized)
            # convert tensor to image
            mask_pred = mask_convert(y_pred[0])
            # convert image to byte
            mask_buffer = io.BytesIO()
            skimage_io.imsave(mask_buffer, mask_pred, format='jpg')
            # Encode the bytes to base64
            mask = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
            masks.append(mask)

    return {
        "mask_image": masks,
    }
