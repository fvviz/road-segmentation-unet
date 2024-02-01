import torchvision
import torch
import numpy as np
from PIL import Image

from model import UNET
from utils import load_checkpoint
import argparse

def process_image(image_path, target_size= (256, 256)):
    resize_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(target_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        )
        ])
    
    uneven_image = Image.open(image_path)
    uneven_image = uneven_image.convert("RGB")
    even_image = resize_transform(uneven_image)
    return even_image

def create_mask(image_path, out_mask_path, device='cpu', checkpoint_path = 'checkpoints/best_checkpoint.pth.tar'):
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze(0)

    model = UNET()
    if device!= 'gpu':
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        mask = torch.sigmoid(model(img_tensor))
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