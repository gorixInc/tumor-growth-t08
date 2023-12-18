from data.data_preprocessing import dice_coefficient
import torch.nn as nn
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

def dice_loss(outputs, masks): 
    return -1 * dice_coefficient(outputs, masks).mean()

def dice_loss2(outputs, inputs):
    intersect = torch.sum(outputs*inputs)
    fsum = torch.sum(outputs)
    ssum = torch.sum(inputs)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = torch.mean(dice) 
    return -dice    

def fit_model(model, train_loader, num_epochs, device, verbose=True, optimizer=None, lr=1e-4, 
              loss_func=None, softmax=False):
    if optimizer is None:    
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_func is None:
        loss_func = dice_loss2
    
    for epoch in range(num_epochs):
        model.train()
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.float().to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            center_crop = transforms.CenterCrop((outputs.shape[-2], outputs.shape[-1]))
            resized_masks = center_crop(masks)
            
            if softmax: 
                resized_masks = nn.Softmax2d()(resized_masks)

            loss = loss_func(outputs, resized_masks)
            loss.backward()
            optimizer.step()

            if verbose and i % 1 == 0:
                print(f'Epoch : {epoch} [{i * len(images)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss:.6f}')

class DoubleConv(nn.Module):
    def __init__(self, in_filters, out_filters, device):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=3, padding='valid', device=device, padding_mode='reflect'),
            nn.BatchNorm2d(out_filters, device=device),
            nn.ReLU(), 
            nn.Conv2d(out_filters, out_filters, kernel_size=3, padding='valid', device=device, padding_mode='reflect'),
            nn.BatchNorm2d(out_filters, device=device),
            nn.ReLU(), 
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Compress(nn.Module):
    def __init__(self, in_filters, out_filters, device, maxpool=True):
        super().__init__()
        self.maxpool = maxpool
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_filters, out_filters, device)

    def forward(self, x):
        if self.maxpool:
            x = self.pool(x)
        return self.conv(x)

class Expand(nn.Module):
    def __init__(self, in_filters, out_filters, device):
        super().__init__()
        self.upscale = nn.ConvTranspose2d(in_filters, in_filters//2, kernel_size=2, stride=2, device=device)
        #self.upscale = nn.ConvTranspose2d(in_filters, in_filters, kernel_size=2, stride=2, device=device)
        self.conv = DoubleConv(in_filters, out_filters, device)

    def forward(self, x_from_down, x_current):
        x_current = self.upscale(x_current)
        delta = [x1_size - x4_size for x1_size, x4_size in zip(x_from_down.size()[2:], x_current.size()[2:])]
        crop_x_from_down = x_from_down[:, :, delta[0]//2:x_from_down.size(2)-delta[0]//2, delta[1]//2:x_from_down.size(3)-delta[1]//2]
        out = torch.cat((x_current, crop_x_from_down), dim=1)
        return self.conv(out)

class OutConv(nn.Module):
    def __init__(self, in_filters, out_filters, device):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=1, device=device, padding_mode='reflect')

    def forward(self, x):
        x = self.conv(x)
        return x

# Original unet https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet
class Unet2D_v2(nn.Module):
    def __init__(self, do_softmax=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(Unet2D_v2, self).__init__()
        self.do_softmax = do_softmax
        self.input_conv = DoubleConv(1, 256, self.device)
        self.comp1 = Compress(256, 512, self.device)
        self.comp2 = Compress(512, 1024, self.device)
        self.exp1 = Expand(1024, 512, self.device)
        self.exp2 = Expand(512, 256, self.device)
        self.out_conv = OutConv(256, 1, self.device)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.comp1(x1)
        x3 = self.comp2(x2)
        x = self.exp1(x2, x3)
        x = self.exp2(x1, x)
        out = self.out_conv(x)
        if self.do_softmax:
            out = self.softmax(out)
        return out


class ConvStack(nn.Module):
    def __init__(self, in_channels, out_channels, device) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='valid', device=device),
            nn.BatchNorm2d(out_channels, device=device),
            nn.ReLU())

    def forward(self, x):
        return self.f(x)


# Kaggle competition model rewritten and with added batchnorm, main model used, the 'small model'
class Unet_classic(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(Unet_classic, self).__init__()
        n_filters = 128
        self.enc_conv11 = ConvStack(1, n_filters, device=self.device)
        self.enc_conv12 =  ConvStack(n_filters, n_filters, device=self.device)
        self.enc_conv13 =  ConvStack(n_filters, n_filters, device=self.device)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv21 = ConvStack(n_filters, n_filters, device=self.device)
        self.enc_conv22 = ConvStack(n_filters, n_filters, device=self.device)
        self.enc_conv23 = ConvStack(n_filters, n_filters, device=self.device)
        self.enc_conv24 = ConvStack(n_filters, n_filters, device=self.device)

        self.upscale1 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=2, stride=2, device=self.device)

        self.dec_conv1 = ConvStack(2*n_filters, 2*n_filters, device=self.device)
        self.dec_conv2 = ConvStack(2*n_filters, 2*n_filters, device=self.device)
        self.dec_conv3 = ConvStack(2*n_filters, n_filters, device=self.device)

        self.final_conv = nn.Conv2d(n_filters, 1, kernel_size=1, device=self.device)
        self.final_act = nn.ReLU()
    
    def forward(self, x):
        x1 = self.enc_conv11(x)
        x1 = self.enc_conv12(x1)
        x1 = self.enc_conv13(x1)

        x2 = self.pool1(x1)
        x2 = self.enc_conv21(x2)
        x2 = self.enc_conv22(x2)
        x2 = self.enc_conv23(x2)
        x2 = self.enc_conv24(x2)

        x3 = self.upscale1(x2)
        delta = [x1_size - x3_size for x1_size, x3_size in zip(x1.size()[2:], x3.size()[2:])]
        crop_x1 = x1[:, :, delta[0]//2:x1.size(2)-delta[0]//2, delta[1]//2:x1.size(3)-delta[1]//2]
        x3 = torch.cat((x3, crop_x1), dim=1)

        x3 = self.dec_conv1(x3)
        x3 = self.dec_conv2(x3)
        x3 = self.dec_conv3(x3)

        x_out = self.final_conv(x3)
        x_out = self.final_act(x_out)

        return x_out

# First version of the Kaggle competition model based on this: https://eliasvansteenkiste.github.io/machine%20learning/lung-cancer-pred/
class UNet2D(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(UNet2D, self).__init__()
        n_filters = 128
        # Convolutional layers with kernel size 3 and no padding (valid)
        # Encoder
        #self.batchnorm_1 = nn.BatchNorm2d(1, device=self.device)
        self.enc_conv1_1 = nn.Conv2d(1, n_filters, kernel_size=3, padding='valid', device=self.device)
        self.enc_conv1_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='valid', device=self.device)
        self.enc_conv1_3 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='valid', device=self.device)

        # Max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Middle encoder layers
        self.encode_conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='valid', device=self.device)
        self.encode_conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='valid', device=self.device)
        self.encode_conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='valid', device=self.device)
        self.encode_conv4 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='valid', device=self.device)

        # Dropout
        self.dropout1 = nn.Dropout()

        # Upscaling
        self.upscale1 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=2, stride=2, device=self.device)

        # Concatenation and Expansion
        self.expand_conv1_1 = nn.Conv2d(2 * n_filters, 2 * n_filters, kernel_size=3, padding='valid', device=self.device)
        self.expand_conv1_2 = nn.Conv2d(2 * n_filters, 2 * n_filters, kernel_size=3, padding='valid', device=self.device)
        self.expand_conv1_3 = nn.Conv2d(2 * n_filters, n_filters, kernel_size=3, padding='valid', device=self.device)

        # Final convolutional layer
        self.final_conv = nn.Conv2d(n_filters, 1, kernel_size=1, device=self.device)

    def forward(self, x):
        # Encoder
        x = x.to(self.device)
        x1 = nn.ReLU()(self.enc_conv1_1(x))
        x1 = nn.ReLU()(self.enc_conv1_2(x1))
        x1 = nn.ReLU()(self.enc_conv1_3(x1))
        x2 = self.pool1(x1)

        # Middle encoder
        x2 = nn.ReLU()(self.encode_conv1(x2))
        x2 = nn.ReLU()(self.encode_conv2(x2))
        x2 = nn.ReLU()(self.encode_conv3(x2))
        x2 = nn.ReLU()(self.encode_conv4(x2))

        # Dropout
        #x2 = self.dropout1(x2)

        # Upscale
        x3 = self.upscale1(x2)
        # Concatenation
        delta = [x1_size - x3_size for x1_size, x3_size in zip(x1.size()[2:], x3.size()[2:])]
        crop_x1 = x1[:, :, delta[0]//2:x1.size(2)-delta[0]//2, delta[1]//2:x1.size(3)-delta[1]//2]
        x3 = torch.cat((x3, crop_x1), dim=1)
        # Expansion
        x3 = nn.ReLU()(self.expand_conv1_1(x3))
        x3 = nn.ReLU()(self.expand_conv1_2(x3))
        x3 = nn.ReLU()(self.expand_conv1_3(x3))

        # Output
        x_out = self.final_conv(x3)
        
        return x_out
    
# Predict on larger image using tiling
def predict(model, X, device, patch_size, encoder_size, mid_size, decoder_size):
    model.eval()
    predicted_scans = []
    stride = patch_size - 2 * encoder_size - 4 * mid_size - 2 * decoder_size
    compression = encoder_size + 2 * mid_size + decoder_size

    for image in X:
        image = torch.from_numpy(image[np.newaxis, np.newaxis, ...]).float()
        H, W = image.shape[-2], image.shape[-1]
        output_scan = torch.zeros_like(image)

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                if y + patch_size > H or x + patch_size > W:
                    continue

                patch = image[:, :, y:y+patch_size, x:x+patch_size]
                patch = patch.to(device)
                output_patch = model(patch)

                output_scan[:, :, y+compression:y+compression+stride, x+compression:x+compression+stride] = output_patch.unsqueeze(0).unsqueeze(0)

        predicted_scans.append(output_scan)

    return torch.cat(predicted_scans, dim=0)  