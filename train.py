import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


# ========================================
#  Dataset (returns occluded, original, mask)
# ========================================
class ImageCompletionDataset(Dataset):
    def __init__(self, occluded_dir, original_dir, transform=None, mask_thresh=0.03):
        self.occluded_dir = occluded_dir
        self.original_dir = original_dir
        self.images = sorted(os.listdir(occluded_dir))
        self.transform = transform
        self.mask_thresh = mask_thresh

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        occluded_path = os.path.join(self.occluded_dir, self.images[idx])
        original_filename = self.images[idx].replace("occluded_", "")
        original_path = os.path.join(self.original_dir, original_filename)

        occluded = Image.open(occluded_path).convert("RGB")
        original = Image.open(original_path).convert("RGB")

        if self.transform:
            occluded = self.transform(occluded)
            original = self.transform(original)

        diff = torch.abs(original - occluded).sum(dim=0, keepdim=True)
        mask = (diff > self.mask_thresh).float()

        return occluded, original, mask



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))


class GatedPixelRNNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_f = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_g = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn_f = nn.BatchNorm2d(channels)
        self.bn_g = nn.BatchNorm2d(channels)

    def forward(self, x):
        f = torch.tanh(self.bn_f(self.conv_f(x)))
        g = torch.sigmoid(self.bn_g(self.conv_g(x)))
        return f * g


# ========================================
#  ConvLSTM Cell
# ========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return h, c


# ========================================
#  SSIM Implementation
# ========================================
def gaussian_window(window_size, sigma, channel):
    coords = torch.arange(window_size).float() - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = g.unsqueeze(1) * g.unsqueeze(0)
    w = w.unsqueeze(0).unsqueeze(0)
    w = w.repeat(channel, 1, 1, 1)
    return w


def ssim(img1, img2, window_size=7, sigma=1.5, data_range=1.0, eps=1e-6):
    B, C, H, W = img1.shape
    device = img1.device
    window = gaussian_window(window_size, sigma, C).to(device)

    padding = window_size // 2
    mu1 = nn.functional.conv2d(img1, window, groups=C, padding=padding)
    mu2 = nn.functional.conv2d(img2, window, groups=C, padding=padding)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(img1 * img1, window, groups=C, padding=padding) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, groups=C, padding=padding) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, groups=C, padding=padding) - mu1_mu2

    L = data_range
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps
    )
    return ssim_map.mean([1, 2, 3]).mean()


# ========================================
#  PixelRNN with ConvLSTM
# ========================================
class PixelRNN(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=48, num_layers=3):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, 7, padding=3)
        self.layers = nn.ModuleList([GatedPixelRNNLayer(hidden_channels) for _ in range(num_layers)])
        self.convlstm = ConvLSTMCell(hidden_channels, hidden_channels)
        self.res_blocks = nn.Sequential(ResidualBlock(hidden_channels), ResidualBlock(hidden_channels))
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, img_rgb, mask):
        x_in = torch.cat([img_rgb, mask], dim=1)
        B, _, H, W = x_in.shape
        device = x_in.device

        x = self.input_conv(x_in)
        skip = x
        h, c = self.convlstm.init_hidden(B, (H, W), device)

        for layer in self.layers:
            x = layer(x)
            h, c = self.convlstm(x, (h, c))
            x = x + h + skip

        x = self.res_blocks(x)
        out = self.output_conv(x)
        return out


# ========================================
#  Training loop with LOGGING
# ========================================
def train(
    occluded_dir="dataset/train/occluded_images",
    original_dir="dataset/train/original_images",
    save_dir="checkpoints",
    epochs=60,
    batch_size=4,
    lr=1e-4,
    hidden_channels=48,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = ImageCompletionDataset(occluded_dir, original_dir, transform=transform, mask_thresh=0.03)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = PixelRNN(in_channels=4, hidden_channels=hidden_channels, num_layers=3).to(device)
    l1 = nn.L1Loss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

    log_path = os.path.join(save_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write("Epoch\tLR\tL1_Loss\tSSIM\tTotal_Loss\n")

    w_l1 = 0.8
    w_ssim = 0.2

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for occluded, original, mask in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            occluded = occluded.to(device)
            original = original.to(device)
            mask = mask.to(device)

            pred = model(occluded, mask)
            comp = occluded * (1.0 - mask) + pred * mask

            pred_masked = pred * mask
            original_masked = original * mask

            loss_l1 = l1(pred_masked, original_masked)
            ssim_val = ssim(pred_masked + (1.0 - mask) * original, original)
            loss_ssim = 1.0 - ssim_val
            loss = w_l1 * loss_l1 + w_ssim * loss_ssim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.6f} | L1: {loss_l1.item():.6f} | SSIM: {ssim_val.item():.6f}")

        # Log the metrics
        with open(log_path, "a") as f:
            f.write(f"{epoch}\t{lr_current:.6f}\t{loss_l1.item():.6f}\t{ssim_val.item():.6f}\t{avg_loss:.6f}\n")

        # Save checkpoints and samples
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch}.pth"))
            model.eval()
            with torch.no_grad():
                n = min(4, occluded.size(0))
                oc = occluded[:n].cpu()
                cm = comp[:n].cpu()
                orr = original[:n].cpu()
                sample = torch.cat([oc, cm, orr], dim=0)
                save_image(sample, os.path.join(save_dir, "samples", f"sample_epoch{epoch}.png"), nrow=n)
            model.train()
            print(f"Saved checkpoint and sample for epoch {epoch}")

    print("Training finished.")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    train(
        occluded_dir="dataset/train/occluded_images",
        original_dir="dataset/train/original_images",
        save_dir="checkpoints",
        epochs=200,
        batch_size=4,
        lr=1e-4,
        hidden_channels=48
    )
