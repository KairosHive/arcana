# Arcana

**Arcana** is a web-based tool to **explore, search, and generate stories in the latent space of large image databases** using state-of-the-art AI models.  
It allows you to interactively search by prompt, visualize clusters, and assemble visual stories—ideal for researchers, artists, and creative technologists.

---

## Features

- **Interactive latent space visualization** (2D or 3D)
- **Prompt-based image search** with CLIP embeddings
- **Story mode:** generate sequences of images from scene descriptions
- **Save selected images and stories** in organized output folders
- Fast thumbnail display for efficient navigation
- Custom dataset selection

---

## Demo - search mode

![image](https://github.com/user-attachments/assets/dd46c1b2-d8db-4417-b173-a9872e01a927)

## Demo - story mode

![image](https://github.com/user-attachments/assets/9977b27d-501e-49ac-9ebc-60bb8d42a467)

---

## Quick Start

### 1. **Clone the repository**

```bash
git clone https://github.com/yourusername/arcana.git
cd arcana
```

### 2. **Install dependencies**

We recommend [conda](https://docs.conda.io/en/latest/miniconda.html) or [virtualenv](https://docs.python.org/3/library/venv.html):

```bash
conda create -n arcana python=3.10
conda activate arcana
pip install -e .
```

### 3. **Prepare your images**

Place your images (jpg, png) inside a subfolder inside the `images/` folder, e.g.:

```
project_root/
  images/
    AntarticaTrip/
      your_images1.png
      your_images2.png
  arcana/
    ...
```

### 4. **Build your latent space and index**

Generate CLIP features, a latent space embedding, and an index for your image folder:

```bash
arcana-build-latent --imgs_path AntarticaTrip --name Antartica --n_components 2
```
- `--imgs_path` is your image folder
- `--name` is the dataset name (used as a key)
- `--n_components` is 2 or 3 (for 2D or 3D visualization)

> This will create all necessary files in `arcana/databases/` and `arcana/latents/`.

### 5. **Run the app**

```bash
arcana
```

The app will launch at `http://127.0.0.1:8050/` (visit in your browser).

---

## Saving and Output

- **Selected images** are saved to `arcana/output/selections/<your-session>/`
- **Stories** (with images and scene descriptions) are saved to `arcana/output/stories/<your-story>/`
