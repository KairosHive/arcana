import math
import numbers
import pickle
from glob import glob
from threading import Thread

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import (
    InterpolationMode,
    gaussian_blur,
    resize,
    rotate,
)
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from usearch.index import Index

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import argparse
import os

# Always use folders INSIDE the script's directory
script_root = os.path.dirname(os.path.abspath(__file__))
IMAGES_ROOT = os.path.abspath(os.path.join(script_root, "..", "images"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

db_dir = os.path.join(script_root, "databases")
latents_dir = os.path.join(script_root, "latents")
os.makedirs(db_dir, exist_ok=True)
os.makedirs(latents_dir, exist_ok=True)


torch.set_grad_enabled(False)

model = CLIPModel.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    device_map="cuda",
    torch_dtype=torch.float16,
)
# model = CLIPModel.from_pretrained(
#     "openai/clip-vit-large-patch14", device_map="cuda"  # , torch_dtype=torch.float16
# )
for param in model.parameters():
    param.requires_grad = False
model.eval()
model.to("cuda")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")


def img2vec(image):
    image_features = processor.image_processor(image, return_tensors="pt").pixel_values
    # image_features = (cv2.resize(image, (224, 224)).astype(np.float16) - 127) / 127
    # image_features = torch.from_numpy(image_features).permute(2, 0, 1).unsqueeze(0)

    out = model.get_image_features(image_features.to("cuda"))
    out = out.squeeze().cpu().float().numpy()
    return out


def txt2vec(text):
    text_features = processor.tokenizer(text, return_tensors="pt")
    out = model.get_text_features(text_features.input_ids.to("cuda"))
    out = out.squeeze().cpu().float().numpy()
    return out


# Add this at the top of your db script
IMAGES_ROOT = os.path.abspath(os.path.join(script_root, "..", "images"))


def build(glob_path, index_path, batch_size=32):
    # recursive=True pairs with ** in glob_path
    paths = glob(glob_path, recursive=True)

    # Keep only files with allowed image extensions
    image_paths = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in IMAGE_EXTS:
            image_paths.append(p)

    print(f"[INFO] Found {len(image_paths)} images to index.")

    index = Index(ndim=1024, metric="cos")
    idx2path = {}

    for batch_start in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[batch_start : batch_start + batch_size]

        # Read with OpenCV; skip unreadable files quietly
        rgb_images = []
        ok_paths = []
        for path in batch_paths:
            bgr = cv2.imread(path)
            if bgr is None:
                print(f"[WARN] Could not read image, skipping: {path}")
                continue
            rgb_images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ok_paths.append(path)

        if not ok_paths:
            continue

        image_inputs = processor(images=rgb_images, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad():
            batch_vecs = model.get_image_features(image_inputs).cpu().float().numpy()

        for i, vec in enumerate(batch_vecs):
            global_idx = batch_start + i
            index.add(global_idx, vec)
            # Store ABSOLUTE path so DBs work no matter where images live
            idx2path[global_idx] = os.path.abspath(ok_paths[i])

    with open(index_path, "wb") as f:
        pickle.dump((index.save(), idx2path), f)

    return index, idx2path


def search(index, idx2path, query):
    vec = txt2vec(query)
    idxs = index.search(vec, 4, exact=True)
    for i, idx in enumerate(idxs):
        plt.subplot(2, 2, i + 1)
        plt.title(f"index {idx.key}, distance {idx.distance:.3f}")
        plt.imshow(cv2.cvtColor(cv2.imread(idx2path[idx.key]), cv2.COLOR_BGR2RGB))
        plt.axis("off")
    plt.show()


def run_search():
    with open("index_laion.pkl", "rb") as f:
        index, idx2path = pickle.load(f)
    index = Index.restore(index)

    while True:
        query = input("Enter a query: ")
        search(index, idx2path, query)


def run_interpolation(steps=100):
    with open("index_all.pkl", "rb") as f:
        index, idx2path = pickle.load(f)
    index = Index.restore(index)

    query1 = input("Enter query 1: ")
    query2 = input("Enter query 2: ")

    vec1 = txt2vec(query1)
    vec2 = txt2vec(query2)

    imgs = []
    indexes = []
    for i in range(steps):
        vec = vec1 * (1 - i / steps) + vec2 * (i / steps)
        idx = index.search(vec, 1, exact=True)[0].key
        if idx in indexes:
            continue
        indexes.append(idx)
        imgs.append(cv2.imread(idx2path[idx]))

        # cv2.imshow("interpolation", cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2RGB))
        # cv2.waitKey(100)

    # for i, img in enumerate(imgs):
    #     plt.subplot(1, steps, i + 1)
    #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     plt.axis("off")
    # plt.show()

    idx = 0
    direction = 1
    while True:
        cv2.imshow("interpolation", cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1000) & 0xFF == ord("q"):
            break

        idx += direction
        if idx >= len(imgs):
            idx = len(imgs) - 1
            direction = -1
        elif idx < 0:
            idx = 0
            direction = 1


def latent_space(index, idx2path, n_components=2):

    vecs = []
    paths = []
    for i in tqdm(idx2path.keys()):
        vecs.append(index.get(i))
        # ensure absolute (older DBs might be relative if you had them)
        p = idx2path[i]
        paths.append(p if os.path.isabs(p) else os.path.abspath(os.path.join(IMAGES_ROOT, p)))
    vecs = np.array(vecs)
    perplexity = min(30, max(5, math.ceil(len(vecs) / 10)))  # Adjust perplexity based on dataset size
    if n_components == 2:
        tsne = TSNE(n_components=2, n_jobs=-1, perplexity=perplexity)  # Ensure 2D embeddings
        vecs = tsne.fit_transform(vecs)

    elif n_components == 3:
        tsne = TSNE(n_components=3, n_jobs=-1)
        vecs = tsne.fit_transform(vecs)
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=10, random_state=0).fit(vecs)
    labels = kmeans.labels_

    return vecs, paths, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Latent Space Builder for Image Datasets")
    parser.add_argument("--imgs_path", type=str, required=True, help="Path to your image folder")
    parser.add_argument("--name", type=str, required=True, help="Project name for saving index and embeddings")
    parser.add_argument(
        "--n_components", type=int, default=2, choices=[2, 3], help="Number of latent space dimensions (2 or 3)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    imgs_path = args.imgs_path  # may be a folder under images/ or an absolute folder
    full_imgs_path = imgs_path if os.path.isabs(imgs_path) else os.path.join(IMAGES_ROOT, imgs_path)

    # NEW: if a directory was provided, turn it into a recursive glob
    glob_arg = os.path.join(full_imgs_path, "**", "*") if os.path.isdir(full_imgs_path) else full_imgs_path

    name = args.name
    n_components = args.n_components

    index_name = os.path.join(db_dir, f"index_{name}.pkl")
    latent_name = os.path.join(latents_dir, f"latent_space_{name}_{n_components}d.pkl")
    print("path to index:", index_name)
    print("path to latent space:", latent_name)
    print("images path (search):", glob_arg)

    index, idx2path = build(glob_arg, index_name)  # << use glob_arg
    vecs, paths, labels = latent_space(index, idx2path, n_components=n_components)

    if n_components == 2:
        df = pd.DataFrame(vecs, columns=["x", "y"])
    elif n_components == 3:
        df = pd.DataFrame(vecs, columns=["x", "y", "z"])
    df["path"] = paths
    df["label"] = labels
    df.to_pickle(latent_name)


if __name__ == "__main__":
    main()
