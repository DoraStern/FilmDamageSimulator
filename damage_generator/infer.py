import torch
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
from train_restoration import UNet


def restore(damaged_path, checkpoint_path, output_path, mask_path=None, size=256, device='cuda', blind=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    in_ch = 3 if blind else 4
    model = UNet(in_ch=in_ch).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    damaged = tf(Image.open(damaged_path).convert('RGB'))

    if blind:
        inp = damaged.unsqueeze(0).to(device)
    else:
        if mask_path:
            mask = tf(Image.open(mask_path).convert('L'))
        else:
            mask = torch.zeros(1, size, size)
        inp = torch.cat([damaged, mask], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)

    out_img = out.squeeze(0).cpu().permute(1, 2, 0).numpy()
    orig_size = Image.open(damaged_path).size
    result = Image.fromarray((out_img * 255).astype(np.uint8)).resize(orig_size, Image.LANCZOS)
    result.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      required=True,  help='damaged image path')
    parser.add_argument('--checkpoint', required=True,  help='path to .pth checkpoint')
    parser.add_argument('--output',     required=True,  help='output image path')
    parser.add_argument('--mask',       default=None,   help='mask image path (optional, masked model only)')
    parser.add_argument('--blind',      action='store_true', help='use blind model (no mask, 3ch input)')
    parser.add_argument('--size',       type=int, default=256)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    restore(args.input, args.checkpoint, args.output, args.mask, args.size, args.device, args.blind)
