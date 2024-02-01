import torchvision
import torch
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UNET
import argparse

def process_image(image_path, target_size= (256, 256)):
    val_transforms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    uneven_image = Image.open(image_path)
    even_image = val_transforms(uneven_image)
    return even_image

def create_mask(image_path, out_mask_path, device='cpu'):
    img_tensor = process_image(image_path)
    img_tensor.to(device)

    model = UNET().to(device)

    model.eval()
    with torch.no_grad():
        mask_raw = model.predict(img_tensor)
        mask = torch.sigmoid(mask_raw)
        mask = (mask>0.5).float()
        torchvision.utils.save_image(mask, out_mask_path)
    model.train()
    print(f"Mask generated successfully at {out_mask_path} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate mask from an input image using UNET model.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('out_mask_path', type=str, help='Path to save the generated mask image')
    parser.add_argument('--device', type=str, default='cpu', help='Device for processing (default: cpu)')

    args = parser.parse_args()

    create_mask(args.image_path, args.out_mask_path, args.device)