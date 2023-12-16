# %%
import numpy as np
import torch
from torch.utils.data import DataLoader 
from torchvision import transforms
from model import UNet2D, Unet2D_v2
from data_preprocessing import MRIScansPatchDataset
import matplotlib.pyplot as plt
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# %%
def train_model(model):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {DEVICE}.")
    BATCH_SIZE = 32
    PATCH_SIZE = 100
    TRAIN_EPOCHS = 5

    X = []
    y = []
    for i in range(100):
        X.append(np.load(f'./2d_dataset/data/Xy_{i}.npy')[0])
        y.append(np.load(f'./2d_dataset/data/Xy_{i}.npy')[1]) 
    
    dataset = MRIScansPatchDataset(X,
                                y,
                                patch_size=PATCH_SIZE)
    train_loader = DataLoader(dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    
    model.fit(train_loader,
            num_epochs=TRAIN_EPOCHS,
            device=DEVICE,
            patch_size=PATCH_SIZE,
            lr=1e-4)
    return model
    
if __name__ == "__main__":
    model = Unet2D_v2()
    train_model(model)
# %%
X_raw = np.load(f'./2d_dataset/data/Xy_{20}.npy')[0]
X = X_raw[np.newaxis, np.newaxis, ...]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output = model.forward(torch.tensor(X).to(DEVICE).float()).cpu()
segm = output.detach().numpy()[0][0]
fig, ax = plt.subplots(1, 2)
ax[0].imshow(X_raw)
segm[np.where(segm < 0.5)] = 0
ax[1].imshow(segm)
fig.tight_layout()
# %%
print(np.sum(segm>0))
# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = []
y = []
for i in range(4):
    X.append(np.load(f'./2d_dataset/data/Xy_{i}.npy')[0])
    y.append(np.load(f'./2d_dataset/data/Xy_{i}.npy')[1]) 

dataset = MRIScansPatchDataset(X,
                            y,
                            patch_size=96)
train_loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True)
for i, (images, masks) in enumerate(train_loader):
    #images, masks = images, masks
    images, masks = images, masks
    images_inp = images.float().to(DEVICE)
    output = model.forward(images_inp).cpu()
    segm = output.detach().numpy()[0][0]
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(segm)
    ax[0].imshow(images[0][0])
    ax[1].imshow(masks[0][0])
    ax[2].imshow(segm)        
# %%
