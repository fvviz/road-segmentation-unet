import torchvision
import torch
import os
import shutil
import splitfolders
import tqdm

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="mps"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

def organise_data(input_dir = 'road-detection/train', split_ratio =(.8, 0.1,0.1) ):
    output_sat_dir = os.path.join(input_dir, 'sat')
    output_mask_dir = os.path.join(input_dir, 'mask')

    os.makedirs(output_sat_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)


    for filename in os.listdir(input_dir):
        if filename.endswith('sat.jpg'):
            shutil.move(os.path.join(input_dir, filename), os.path.join(output_sat_dir, filename))
        elif filename.endswith('mask.png'):
            shutil.move(os.path.join(input_dir, filename), os.path.join(output_mask_dir, filename))
    splitfolders.ratio('road-detection/train', output="road-detection/organised_data", seed=1337, ratio=split_ratio)


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for i, (data, targets) in enumerate(loop):
        data = data.to(device='cuda')
        targets = targets.float().unsqueeze(1).to(device='cuda')


        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
