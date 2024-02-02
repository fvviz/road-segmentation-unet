import torch
import torch.nn.functional as F
from PIL import Image
import argparse

from torchvision import transforms
import torchvision
from model import UNET

def sliding_window_inference(model, input_tensor, window_size, stride, threshold):
    _, _, height, width = input_tensor.size()
    result_tensor = torch.zeros((1, 1, height, width), device=input_tensor.device)
    count_tensor = torch.zeros((1, 1, height, width), device=input_tensor.device)

    model.eval()
    for h in range(0, height - window_size[2] + 1, stride):
        for w in range(0, width - window_size[3] + 1, stride):
            patch = input_tensor[:, :, h:h+window_size[2], w:w+window_size[3]]


            with torch.no_grad():
                output_patch = torch.sigmoid(model(patch))
                output_patch =(output_patch>threshold).float()

    
            result_tensor[:, :, h:h+window_size[2], w:w+window_size[3]] += output_patch
            count_tensor[:, :, h:h+window_size[2], w:w+window_size[3]] += 1

    result_tensor /= count_tensor
    model.train()

    return result_tensor

def run_sliding_window_pil(image, threshold, window_pixels, stride=64):
    model = UNET()
    checkpoint = torch.load('checkpoints/epoch_3_checkpoint.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])

    input_image = transform(image).unsqueeze(0)

    window_size = (1, 3, window_pixels, window_pixels)
    print("running sliding window")
    output = sliding_window_inference(model, input_image, window_size, stride, threshold)

    normalized_output = output
    denormalized_output = normalized_output * torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1) + torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1)
    
    # Converting torch tensor to PIL Image for Gradio compatibility
    denormalized_output_pil = transforms.ToPILImage()(denormalized_output.squeeze(0))

    return denormalized_output_pil

def run_sliding_window(image_dir, output_dir, threshold):
    model = UNET()
    checkpoint = torch.load('checkpoints/epoch_3_checkpoint.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    print("loaded checkpoints")

    img = Image.open(image_dir)
    img = img.convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])

    input_image = transform(img).unsqueeze(0)

    window_size = (1, 3, 256, 256)
    stride = 64
    output = sliding_window_inference(model, input_image, window_size, stride, threshold)

    normalized_output = output 
    denormalized_output = normalized_output * torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1) + torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1)
    torchvision.utils.save_image(denormalized_output, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Run sliding window inference on an image.')
    parser.add_argument('image_dir', type=str, help='Path to the input image directory.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory.')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')

    args = parser.parse_args()

    run_sliding_window(args.image_dir, args.output_dir, args.threshold)

if __name__ == "__main__":
    main()