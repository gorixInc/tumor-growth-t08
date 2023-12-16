# %%
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UNet2D
from data_preprocessing import MRIScansPatchDataset
# %%
def train_model(model):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {DEVICE}.")
    BATCH_SIZE = 2
    PATCH_SIZE = 128
    TRAIN_EPOCHS = 40

    X = []
    y = []
    for i in range(6):
        X.append(np.load(f'./2d_dataset_example/data/Xy_{i}.npy')[0])
        y.append(np.load(f'./2d_dataset_example/data/Xy_{i}.npy')[1]) 

    dataset = MRIScansPatchDataset(X,
                                y,
                                patch_size=PATCH_SIZE)
    train_loader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    
    model.fit(train_loader,
            num_epochs=TRAIN_EPOCHS,
            device=DEVICE,
            patch_size=PATCH_SIZE)
    return model
    
if __name__ == "__main__":
    model = UNet2D()
    train_model(model)
# %%

# %%
