# %%
import numpy as np
import torch
from torch.utils.data import DataLoader 
from torchvision import transforms
from model import UNet2D, Unet2D_v2, Unet_classic, fit_model
from data.data_preprocessing import MRIScansPatchDataset
import matplotlib.pyplot as plt
from torch.nn.functional import pad
import os 
from glob import glob
from model import predict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# %%
dataset_path = './2d_dataset_norm/*'
# %%
random_inds = np.random.randint(0, 1200, size=200)
# %%
def train_model(model):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {DEVICE}.")
    BATCH_SIZE = 8
    PATCH_SIZE = 128
    TRAIN_EPOCHS = 1

    X = []
    y = []
    for path in np.array(glob(dataset_path))[random_inds]:
        X.append(np.load(path)[0])
        y.append(np.load(path)[1]) 
    
    dataset = MRIScansPatchDataset(X, y, patch_size=PATCH_SIZE)
    train_loader = DataLoader(dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    
    fit_model(model, train_loader,
            num_epochs=TRAIN_EPOCHS,
            device=DEVICE,
            lr=1e-4)
    return model
    
if __name__ == "__main__":
    model = Unet_classic()
    train_model(model)

# %%
random_inds2 = np.random.randint(0, 1200, size=200)
random_inds2 = [i for i in random_inds2 if i not in random_inds]
crop_x = 200
crop_y = 20
for i, path in enumerate(np.array(glob(dataset_path))[random_inds2]):

    Xy = np.load(path)
    X_raw = Xy[0]
    y = Xy[1]
    X = X_raw[np.newaxis, ...]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = predict(model, X, DEVICE, 128, 3, 4, 3)
    segm = output.detach().numpy()[0][0]

    diffy = output.shape[-1] - X_raw.shape[-1]
    diffx = output.shape[-2] - X_raw.shape[-2]
    padded_segm = pad(output, (-diffx//2, -diffx//2, -diffy//2, -diffy//2)).detach().numpy()
    segm = padded_segm[0][0]

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(X_raw[crop_x : -crop_x, crop_y : - crop_y])
    ax[1].imshow(y[crop_x : -crop_x, crop_y : - crop_y])
    ax[2].imshow(segm[crop_x : -crop_x, crop_y : - crop_y])
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    ax[0].set_title('Input')
    ax[1].set_title('Label')
    ax[2].set_title('Segmentation')

    fig.tight_layout()
    fig.savefig(f'relu_full/{i}.jpg', dpi=200)
# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = []
y = []
random_inds3 = np.random.randint(0, 1200, size=100)
random_inds3 = [i for i in random_inds3 if i not in random_inds]
for path in np.array(glob(dataset_path))[random_inds3]:
    X.append(np.load(path)[0])
    y.append(np.load(path)[1]) 
    

dataset = MRIScansPatchDataset(X,
                            y,
                            patch_size=128)
train_loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True)
for i, (images, masks) in enumerate(train_loader):
    #images, masks = images, masks
    images, masks = images, masks
    images_inp = images.float().to(DEVICE)
    output = model.forward(images_inp).cpu()

    center_crop = transforms.CenterCrop((output.shape[-2], output.shape[-1]))
    resized_masks = center_crop(masks)
    diffy = output.shape[-1] - images.shape[-1]
    diffx = output.shape[-2] - images.shape[-2]
    padded_segm = pad(output, (-diffx//2, -diffx//2, -diffy//2, -diffy//2)).detach().numpy()
    segm = padded_segm[0][0]
    
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(images[0][0])
    ax[1].imshow(masks[0][0])
    ax[2].imshow(segm)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    ax[0].set_title('Input')
    ax[1].set_title('Label')
    ax[2].set_title('Segmentation')
    fig.savefig(f'relu_patch/{i}.jpg', dpi=200)
# %%
random_inds3 = np.random.randint(0, 1200, size=10)
random_inds3 = [i for i in random_inds3 if i not in random_inds]
for path in np.array(glob(dataset_path))[random_inds3]:
    X.append(np.load(path)[0])
    y.append(np.load(path)[1]) 
dataset = MRIScansPatchDataset(X,
                            y,
                            patch_size=256)
train_loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True)
for i, (images, masks) in enumerate(train_loader):
    #images, masks = images, masks
    images, masks = images, masks

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(images[0][0])
    ax[1].imshow(masks[0][0])
    #ax[1].imshow(segm)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    ax[0].set_title('Input')
    ax[1].set_title('Label')
    #ax[2].set_title('Segmentation')