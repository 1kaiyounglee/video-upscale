## IMPORTS ##
import cv2
import torch
import os
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from tqdm import tqdm

## DEFINITIONS ##

# Config
input_video = '../input.mp4'
output_video = '../output.mp4'
model_path = '../RealESRGAN_x4plus.pth'
scale = 4

# Load model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                num_grow_ch=32, scale=scale)
upsampler = RealESRGANer(
    scale=scale, model_path=model_path, model=model,
    tile=0, half=True, pre_pad=0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Open video
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (w * scale, h * scale))


print(f"Upscaling video ({w} * {h} -> {w*scale} * {h*scale}), total frames: {frame_count}")

for _ in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output, _  = upsampler.enhance(img, outscale=scale)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    out.write(output_bgr)

cap.release()
out.release()

print(f"Done! Saved as: {output_video}")
