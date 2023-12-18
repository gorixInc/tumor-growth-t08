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
        if self.patch_size is not None:
            image_patch, mask_patch = self.extract_patch(image, mask, centroid)
        else:
            image_patch, mask_patch = image, mask

        if self.transform:
            image_patch = self.transform(image_patch)
            mask_patch = self.transform(mask_patch)
        return image_patch[np.newaxis, ...], mask_patch[np.newaxis, ...]

    def extract_patch(self, image, mask, centroid):        
        y, x = int(centroid[0]), int(centroid[1])
        half_patch = self.patch_size // 2
        # Calculate start and end coordinates for y and x        
        start_y = max(y - half_patch, 0)
        end_y = min(y + half_patch, image.shape[0])        
        start_x = max(x - half_patch, 0)
        end_x = min(x + half_patch, image.shape[1])
        # Extract the patches        
        image_patch = image[start_y:end_y, start_x:end_x]
        mask_patch = mask[start_y:end_y, start_x:end_x]
        # Calculate padding to add if the patch is smaller than expected        
        pad_top = max(half_patch - y, 0)
        pad_bottom = max((y + half_patch) - image.shape[0], 0)       
        pad_left = max(half_patch - x, 0)
        pad_right = max((x + half_patch) - image.shape[1], 0)
        # Apply padding        
        image_patch = np.pad(image_patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        mask_patch = np.pad(mask_patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        return image_patch, mask_patch
    
def dice_coefficient(pred, target, smooth=1e-12):
    intersection = (pred * target).sum(axis=(1, 2))
    return (2. * intersection + smooth) / (pred.sum(axis=(1, 2)) + target.sum(axis=(1, 2)) + smooth)