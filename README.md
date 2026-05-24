# Image-to-Image Retrieval System Using FAISS

> Computer Vision · Rice University COMP646 · Aug – Dec 2023

Scalable image retrieval system that extracts feature vectors from 1 million images using a pre-trained **SigLIP** model and evaluates **FAISS** indexes for fast k-nearest-neighbour search. Benchmarks Recall@K and latency across `IndexFlatL2` and `IndexIVFFlat` to quantify the speed/accuracy trade-off.

## Highlights
- **Sub-14ms retrieval latency** at 1M scale using `IndexIVFFlat`
- **Recall@1 > 0.95** on the SBU Captions dataset
- Supports batched feature extraction across multiple GPU nodes (Rice NOTS cluster)
- Configurable via CLI — no hardcoded paths

## Architecture
```
SBU Captions Dataset (1M images)
           │
  ImageFeaturesExtractor.py        ← SigLIP encoder (HuggingFace)
           │                          outputs 100K-row CSV chunks
  IndexCreation.ipynb              ← builds FAISS index per chunk
           │
  Merge100K.ipynb                  ← merges chunks → 1M index file
           │
  ┌────────┴────────┐
  │                 │
LatencyMesaurement  Recall.ipynb   ← benchmark latency + Recall@K
```

## Results
| Index Type | Latency (1M vectors) | Recall@1 |
|---|---|---|
| `IndexFlatL2` (exact) | ~180ms | 1.00 |
| `IndexIVFFlat` (approximate) | **~14ms** | **0.95+** |

---

## Setup & Running

### 1. Install Dependencies
```bash
git clone https://github.com/tsenthil5/COMP-646-Faiss-Project
cd COMP-646-Faiss-Project
pip install -r requirements.txt
```

> Requires Python 3.9+. For GPU support replace `faiss-cpu` with `faiss-gpu` in `requirements.txt`.

---

### 2. Download the Dataset

```bash
# SBU Captions images (~9GB)
curl -L "http://www.cs.rice.edu/~vo9/sbucaptions/sbu_images.tar" -o sbu_images.tar
tar -xf sbu_images.tar
```

---

### 3. Extract Image Features

```bash
python ImageFeaturesExtractor.py \
    --image-dir ./sbu-captions/images \
    --output-path ./features \
    --start-folder 0 \
    --end-folder 9 \
    --batch-size 64
```

This saves CSV files (one per folder range) under `./features/`.

---

### 4. Build the FAISS Index

Open and run `IndexCreation.ipynb` in Jupyter:
```bash
pip install jupyterlab
jupyter lab IndexCreation.ipynb
```
Update the `FEATURES_DIR` and `OUTPUT_DIR` variables at the top of the notebook to point to your local paths.

---

### 5. Merge 100K Indexes → 1M Index

Run `Merge100K.ipynb` — set `INDEX_DIR` to where the per-chunk indexes were saved.

Alternatively, download the pre-built 1M index files directly:
- [IndexFlatL2 (1M)](https://drive.google.com/file/d/1i77SSTKCtDipQaCvr08yCEWgLAJUjYPU/view?usp=share_link)
- [IndexIVFFlat (1M)](https://drive.google.com/file/d/19Ob35sH_iNT8Zqu66mwA44JREj2t1K-4/view?usp=share_link)

---

### 6. Benchmark Latency & Recall

Run `LatencyMesaurement.ipynb` and `Recall.ipynb` — set the index file paths in the first notebook cell.

---

## Code Structure

| File | Purpose |
|---|---|
| `ImageFeaturesExtractor.py` | CLI script — extracts SigLIP feature vectors from images, saves as CSV |
| `IndexCreation.ipynb` | Builds a FAISS index from feature CSVs |
| `Merge100K.ipynb` | Merges per-chunk indexes into a single 1M index |
| `LatencyMesaurement.ipynb` | Benchmarks search latency across index types |
| `Recall.ipynb` | Computes Recall@K (K=1,10,20,...,100) |
| `job.slurm` | SLURM job script for Rice University NOTS cluster |

## Tech Stack
`Python` `PyTorch` `SigLIP` `FAISS` `HuggingFace Transformers` `NumPy` `Pandas` `Jupyter`
