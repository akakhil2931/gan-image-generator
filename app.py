import streamlit as st
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from model import Generator
from io import BytesIO
from PIL import Image

# -----------------------------
# Config (must match training)
# -----------------------------
Z_DIM = 128        # your checkpoint expects 128 (from the mismatch you saw)
IMG_CHANNELS = 3
G_FEAT = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="CIFAR-10 GAN Generator", page_icon="🎨", layout="wide")

# -----------------------------
# Load the trained generator
# -----------------------------
@st.cache_resource
def load_generator():
    gen = Generator(z_dim=Z_DIM, img_channels=IMG_CHANNELS, gfeat=G_FEAT).to(DEVICE)
    state_dict = torch.load("generator.pth", map_location=DEVICE)
    # If you still run into strict errors, change strict=True -> False
    gen.load_state_dict(state_dict, strict=True)
    gen.eval()
    return gen

generator = load_generator()

# -----------------------------
# Helpers
# -----------------------------
def make_grid_image(tensor_batch, nrow=8):
    # tensor_batch in [-1, 1] → grid image as PIL
    grid = vutils.make_grid(tensor_batch, nrow=nrow, normalize=True, value_range=(-1, 1))
    ndarr = (grid.mul(255).clamp(0, 255)).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(ndarr)

def to_pil_single(x):
    # x in [-1, 1] → [0, 1]
    x = (x * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(x)

def generate_images(num_images=16, nrow=8, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(num_images, Z_DIM, 1, 1, device=DEVICE)
    with torch.no_grad():
        fake_images = generator(noise).cpu()
    grid_img = make_grid_image(fake_images, nrow=nrow)
    return grid_img, fake_images  # PIL grid, tensor batch

# -----------------------------
# UI
# -----------------------------
st.title("🎨 CIFAR-10 GAN Image Generator")
st.caption(f"Running on **{DEVICE.upper()}** · DCGAN trained on CIFAR-10 · Latent size **{Z_DIM}**")

st.markdown(
    "Use the sidebar to choose how many images to generate and layout options. "
    "Click **Generate** to produce a fresh batch."
)

# Sidebar controls
st.sidebar.header("⚙️ Controls")
num = st.sidebar.slider("Number of images", 1, 64, 16, step=1)
nrow = st.sidebar.slider("Images per row", 1, 16, 8, step=1)
seed_opt = st.sidebar.text_input("Random seed (optional)", value="")
seed_val = int(seed_opt) if seed_opt.strip().isdigit() else None
generate = st.sidebar.button("🚀 Generate Images")

if generate:
    grid_img, batch = generate_images(num_images=num, nrow=nrow, seed=seed_val)

    # Show grid
    st.subheader("🖼️ Generated Image Grid")
    st.image(grid_img, caption=f"{num} generated CIFAR-10-like images", use_container_width=True)

    # Download grid
    buf = BytesIO()
    grid_img.save(buf, format="PNG")
    st.download_button(
        label="📥 Download Grid (PNG)",
        data=buf.getvalue(),
        file_name="generated.png",
        mime="image/png",
    )

    # Gallery
    st.subheader("🔎 Explore Individual Images")
    cols = st.columns(min(8, num))
    for i, img_tensor in enumerate(batch):
        pil_img = to_pil_single(img_tensor)
        with cols[i % len(cols)]:
            st.image(pil_img, use_container_width=True)
