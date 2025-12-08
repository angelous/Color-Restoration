import gc
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from io import BytesIO
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide", page_title="Image Colorization App")

st.write("## Image Colorization")
st.write(
    "Restore color to black and white images"
)
st.sidebar.write("## Upload and download")

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()

        # Encoder: ResNet18 Pretrained
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Decoder
        self.decoder4 = self._block(512, 256)
        self.decoder3 = self._block(256 + 256, 128)
        self.decoder2 = self._block(128 + 128, 64)
        self.decoder1 = self._block(64 + 64, 32)

        self.final = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d4 = self.decoder4(e5)
        d3 = self.decoder3(torch.cat([d4, e4], dim=1))
        d2 = self.decoder2(torch.cat([d3, e3], dim=1))
        d1 = self.decoder1(torch.cat([d2, e2], dim=1))

        out = self.final(d1)
        out = self.upsample(out)

        return out

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationNet().to(device)
    model_path = 'model_colorization_tuned (2).pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

def colorize_image(image_path):
    # 1. Buka Gambar & Preprocessing
    img = Image.open(image_path).convert("RGB")

    # Simpan ukuran asli untuk resize balik nanti (opsional, biar rapi)
    original_size = img.size

    # Resize ke 256x256 (karena model dilatih di ukuran ini)
    transform = transforms.Resize((256, 256))
    img_resized = transform(img)

    # Konversi ke LAB dan Ambil L
    img_lab = rgb2lab(np.array(img_resized)).astype("float32")
    img_l = img_lab[:, :, 0] / 100.0 # Normalisasi L

    # Siapkan Tensor
    input_l = torch.from_numpy(img_l).unsqueeze(0).unsqueeze(0).float().to(device)

    # 2. Prediksi (Inference)
    with torch.no_grad():
        pred_ab = model(input_l)

    # 3. Post-processing (Gabung L + ab prediksi)
    SATURATION = 1.5
    pred_ab = pred_ab.squeeze().cpu().numpy().transpose((1, 2, 0)) * 128.0 * SATURATION

    # Gabung L asli (yang diresize) dengan ab hasil prediksi
    result_lab = np.zeros((256, 256, 3))
    result_lab[:, :, 0] = img_lab[:, :, 0]
    result_lab[:, :, 1:] = pred_ab

    # Convert balik ke RGB
    result_rgb = lab2rgb(result_lab)

    result_rgb_uint8 = (np.clip(result_rgb, 0, 1) * 255).astype("uint8")
    result_img_pil = Image.fromarray(result_rgb_uint8)
    result_img_original_size = result_img_pil.resize(original_size, Image.BICUBIC)

    del input_l, pred_ab, result_lab, result_rgb, img_lab
    gc.collect()

    return np.array(result_img_original_size) / 255.0

def convert_image_np_to_png(np_img):
    # np_img bentuknya float 0–1 → convert ke uint8
    np_img_uint8 = (np.clip(np_img, 0, 1) * 255).astype("uint8")
    img_pil = Image.fromarray(np_img_uint8)

    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

col1, col2 = st.columns(2)
image_input = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

with st.sidebar.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Large images will be automatically resized
    - Supported formats: PNG, JPG, JPEG
    """)

if image_input is not None:
    # Simpan gambar yang diupload ke disk sementara
    with open("temp_image.png", "wb") as f:
        f.write(image_input.getbuffer())

    # Warna gambar
    colorized_img = colorize_image("temp_image.png")

    # Tampilkan hasil
    with col1:
        st.write("### Original Image")
        st.image(Image.open("temp_image.png"), width=500)

    with col2:
        st.write("### Colorized Image")
        st.image(colorized_img, width=500)
    st.sidebar.download_button(
        "Download Colorized Image",
        convert_image_np_to_png(colorized_img),
        "colorized.png",
        "image/png"
    )

    # Hapus file sementara
    if os.path.exists("temp_image.png"):
        os.remove("temp_image.png")
        
    # Final Garbage Collection
    gc.collect()

else:
    default_image_path = "./image5006.jpg"
    colorized_img = colorize_image(default_image_path)

    with col1:
        st.write("### Original Image")
        st.image(Image.open(default_image_path), width=500)

    with col2:
        st.write("### Colorized Image")
        st.image(colorized_img, width=500)