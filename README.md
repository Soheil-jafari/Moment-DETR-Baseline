# 🎯 Moment-DETR for Custom Datasets  
### Modular Transformer-Based Temporal Video Grounding Pipeline

A clean, production-ready implementation of **Moment-DETR** for language-based temporal video grounding on custom datasets.

Given a natural language query, the model predicts the **[start, end] timestamps** in a video where the described event occurs.

This repository refactors the original Moment-DETR implementation into a modular, reproducible, and extensible pipeline, making it easy to train and evaluate on new datasets.

---

## 📌 Problem Overview

**Task:** Temporal Video Grounding  
**Input:** Video + Natural Language Query  
**Output:** Predicted temporal segment `[t_start, t_end]`

Example:

> Query: *"The surgeon inserts the endoscope."*  
> Model Output: `[12.4s, 18.7s]`

---

## 🏗 Architecture & Pipeline

The training workflow is split into four independent stages:

Preprocessing → Feature Extraction → Training → Evaluation

Each stage is handled by a dedicated script:

| Stage | Script | Description |
|-------|--------|------------|
| Preprocessing | `run_preprocessing.py` | Converts CSV annotations → JSONL |
| Feature Extraction | `run_feature_extraction.py` | Extracts ResNet-50 visual features |
| Training | `run_training.py` | Distributed training with Moment-DETR |
| Evaluation | `run_evaluation.py` | Computes mAP@tIoU & Recall@k |

This modular design allows independent execution of each stage for flexible experimentation.

---

## 🚀 Key Features

- End-to-end temporal grounding pipeline  
- Modular and extensible design  
- ResNet-50 feature backbone  
- Multi-GPU distributed training (`torch.distributed`)  
- Standard evaluation metrics (mAP@tIoU, Recall@k)  
- Plug-and-play dataset configuration  

---

## ⚙️ Installation

We recommend using a dedicated Conda environment.

```bash
# Clone repository
git clone https://github.com/yourname/moment-detr-custom.git
cd moment_detr_baseline

# Create environment
conda create -n momentdetr python=3.9 -y
conda activate momentdetr

# Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

```
📂 Dataset Format

Annotations should be provided as a CSV file:
```bash
video_id,text,start_frame,end_frame
```
Example:
```bash
video_001,"person opens door",120,240
```
🛠️ Usage: Step-by-Step
Step 0: Configure Paths

Define dataset paths inside your config (or directly inside the scripts).
Make sure run_preprocessing.py and run_feature_extraction.py point to the correct locations of your CSV files, videos, and output folders.

Step 1: Preprocess Annotations
```bash
Convert CSV annotations → JSONL format.
python run_preprocessing.py
```

✅ Outputs:
```bash
preprocessed_data/
  ├── train.jsonl
  ├── val.jsonl
  └── test.jsonl
```
Step 2: Extract Visual Features

Extract visual features using ResNet-50.
```bash
python run_feature_extraction.py
```

Resumable (skips completed videos)

Saves features to:
```text
extracted_features_resnet50/
```
Step 3: Train the Model (Multi-GPU)

Train using distributed multi-GPU support:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port 29501 \
  run_training.py
```

Checkpoints saved to:
```text
checkpoints/
```
Adjust --nproc_per_node based on the number of GPUs available.

Step 4: Evaluate the Model

Evaluate a trained checkpoint:
```bash
python run_evaluation.py \
  --resume /path/to/checkpoints/run_name/best_checkpoint.ckpt
```
<p align="center"> <img width="850" alt="Temporal Grounding"
src="docs/timeline.png" /> </p>                     
Results are printed and logged

Metrics include mAP@tIoU and Recall@k

📊 Evaluation Metrics

mAP@tIoU — Mean Average Precision at temporal IoU thresholds

Recall@k — Top-k retrieval accuracy

📚 Citation

This repo adapts from the official Moment-DETR implementation.

If you use this repo, please cite the original paper:
```bibtext
@inproceedings{momentdetr2021,
  title={End-to-End Video Instance Segmentation with Transformers},
  author={Wang, Yuqing and Xu, Zhaoliang and Wang, Xinlong and Li, Chun-Guang and Yao, Yong-Qiang and Li, Yue-Meng and Meng, Gaofeng},
  booktitle={CVPR},
  year={2021}
}
```
