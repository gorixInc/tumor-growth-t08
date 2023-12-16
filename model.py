from data_preprocessing import dice_coefficient
import torch.nn as nn
import torch
from torchvision import transforms

class UNet2D(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        super(UNet2D, self).__init__()
        n_filters = 128

        # Convolutional layers with kernel size 3 and no padding (valid)
        # Encoder
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
        x2 = self.dropout1(x2)

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
        x_out = torch.sigmoid(self.final_conv(x3))
        
        return x_out

    def fit(self, train_loader, num_epochs, device, patch_size, verbose=True):
        optimizer = torch.optim.Adam(self.parameters())
        for epoch in range(num_epochs):
            self.train()
            for i, (images, masks) in enumerate(train_loader):
                images, masks = images.float().to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = self.forward(images)
                
                center_crop = transforms.CenterCrop((outputs.shape[-1], outputs.shape[-1]))
                resized_masks = center_crop(masks)

                loss = -1 * dice_coefficient(outputs, resized_masks).mean()
                loss.backward()
                optimizer.step()

                if verbose and i % 1 == 0:
                    print(f'Epoch : {epoch} [{i * len(images)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss:.6f}')
                    