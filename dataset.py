import os
from PIL import Image
import torch
from torchvision import transforms

# Đường dẫn tới folder chứa ảnh
folder = '/home/nghiapd/Code/DAT_model/Set5/image_SRF_2'

# Lấy danh sách file ảnh GR và LR
gr_files = sorted([f for f in os.listdir(folder) if '_GR' in f])
lr_files = sorted([f for f in os.listdir(folder) if '_LR' in f])

# Hàm chuyển ảnh sang tensor
to_tensor = transforms.ToTensor()

for gr_file, lr_file in zip(gr_files, lr_files):
    gr_path = os.path.join(folder, gr_file)
    lr_path = os.path.join(folder, lr_file)
    gr_img = Image.open(gr_path).convert('L')
    lr_img = Image.open(lr_path).convert('L')
    gr_tensor = to_tensor(gr_img).unsqueeze(0)
    lr_tensor = to_tensor(lr_img).unsqueeze(0)
    print(f"{gr_file}: GR shape {gr_tensor.shape} | {lr_file}: LR shape {lr_tensor.shape}")
    if gr_tensor.shape != lr_tensor.shape:
        print(f"  -> ⚠️ Different sizes!")