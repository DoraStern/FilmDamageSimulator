import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class FilmDamageDataset(Dataset):
    def __init__(self, generated_dir, size=256):
        self.samples = sorted(glob.glob(os.path.join(generated_dir, 'damage_mask_*')))
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        original = self.tf(Image.open(os.path.join(d, 'image.png')).convert('RGB'))
        damaged  = self.tf(Image.open(os.path.join(d, 'damaged.png')).convert('RGB'))
        mask     = self.tf(Image.open(os.path.join(d, 'mask.png')).convert('L'))
        inp = torch.cat([damaged, mask], dim=0)  # 4ch: RGB damaged + mask
        return inp, original


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2)

        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.head = nn.Sequential(nn.Conv2d(features[0], out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.head(x)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return nn.functional.l1_loss(self.features(pred), self.features(target))


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Device: {device}")

    dataset = FilmDamageDataset(args.data_dir, args.size)
    print(f"Samples: {len(dataset)}")
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)

    model  = UNet().to(device)
    optim_ = optim.Adam(model.parameters(), lr=args.lr)
    sched  = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=args.epochs)
    l1     = nn.L1Loss()
    perc   = PerceptualLoss().to(device) if args.perc_weight > 0 else None

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for inp, target in loader:
            inp, target = inp.to(device), target.to(device)
            pred = model(inp)
            loss = l1(pred, target)
            if perc is not None:
                loss = loss + args.perc_weight * perc(pred, target)
            optim_.zero_grad()
            loss.backward()
            optim_.step()
            total_loss += loss.item()

        sched.step()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch:03d}/{args.epochs}  loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if epoch % args.save_every == 0:
            path = os.path.join(args.save_dir, f'restoration_epoch{epoch:03d}.pth')
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optim': optim_.state_dict()}, path)
            print(f"Saved {path}")

    torch.save({'epoch': args.epochs, 'model': model.state_dict()},
               os.path.join(args.save_dir, 'restoration_final.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',    default='generated',   help='path to generated/ folder')
    parser.add_argument('--save-dir',    default='checkpoints', help='where to save .pth files')
    parser.add_argument('--size',        type=int,   default=256)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch-size',  type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--perc-weight', type=float, default=0.1,  help='perceptual loss weight')
    parser.add_argument('--save-every',  type=int,   default=10)
    parser.add_argument('--workers',     type=int,   default=2)
    parser.add_argument('--device',      default='cuda')
    args = parser.parse_args()
    train(args)
