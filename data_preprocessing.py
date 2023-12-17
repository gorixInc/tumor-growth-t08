from torch.utils.data import Dataset
from scipy.ndimage import center_of_mass
import numpy as np

class MRIScansPatchDataset(Dataset):
    def __init__(self, images, masks, patch_size, transform=None):
        self.images = images
        self.masks = masks
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        #print(image.shape, mask.shape)
        centroid = center_of_mass(mask)
        
        image_patch, mask_patch = self.extract_patch(image, mask, centroid)
        if self.transform:
            image_patch = self.transform(image_patch)
            mask_patch = self.transform(mask_patch)
        
        return image_patch[np.newaxis, ...], mask_patch[np.newaxis, ...]

    def extract_patch(self, image, mask, centroid):
        y, x = int(centroid[0]), int(centroid[1])
        half_patch = self.patch_size // 2

        # IF YOU GET THE INDEX OUT OF RANGE ERROR, THEN TRY A SMALLER PATCH SIZE!
        # patch must be a square, so in case of setting boundaries, additional padding will be required, which might reduce the quality of model
        image_patch = image[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
        mask_patch = mask[y-half_patch:y+half_patch, x-half_patch:x+half_patch]

        return image_patch, mask_patch
    
def dice_coefficient(pred, target, smooth=1e-12):
    intersection = (pred * target).sum(axis=(1, 2))
    return (2. * intersection + smooth) / (pred.sum(axis=(1, 2)) + target.sum(axis=(1, 2)) + smooth)