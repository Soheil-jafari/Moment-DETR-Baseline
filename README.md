🎯 Moment-DETR for Custom Datasets
Modular Transformer-Based Temporal Video Grounding Pipeline

A clean, production-ready implementation of Moment-DETR for language-based temporal video grounding on custom datasets.

Given a natural language query, the model predicts the [start, end] timestamps in a video where the described event occurs.

This repository refactors the original Moment-DETR implementation into a modular, reproducible, and extensible pipeline, making it easy to train and evaluate on new datasets.

📌 Problem Overview

Task: Temporal Video Grounding
Input: Video + Natural Language Query
Output: Predicted temporal segment [t_start, t_end]

Example:

Query: “The surgeon inserts the endoscope.”
Model Output: [12.4s, 18.7s]

🏗 Architecture & Pipeline

The training workflow is split into four independent stages:

Preprocessing → Feature Extraction → Training → Evaluation


Each stage is handled by a dedicated script:

Stage	Script	Description
📑 Preprocessing	run_preprocessing.py	Converts CSV annotations → JSONL
🖼 Feature Extraction	run_feature_extraction.py	Extracts ResNet-50 visual features
🧠 Training	run_training.py	Distributed training with Moment-DETR
📊 Evaluation	run_evaluation.py	Computes mAP@tIoU & Recall@k

This modular design allows independent execution of each stage for flexible experimentation.

🚀 Key Features

✅ End-to-end temporal grounding pipeline

✅ Modular and easily extensible design

✅ ResNet-50 feature backbone (efficient + stable)

✅ Multi-GPU distributed training (torch.distributed)

✅ Standard evaluation metrics (mAP@tIoU, Recall@k)

✅ Plug-and-play dataset configuration

⚙️ Installation

We recommend using a dedicated Conda environment.

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

📂 Dataset Format

Annotations should be provided as a CSV file:

video_id,text,start_frame,end_frame


Example:

video_001,"person opens door",120,240

🛠 Usage
1️⃣ Preprocess Annotations
python run_preprocessing.py


Outputs:

preprocessed_data/
  ├── train.jsonl
  ├── val.jsonl
  └── test.jsonl

2️⃣ Extract Visual Features
python run_feature_extraction.py


Uses ResNet-50 backbone

Automatically skips processed videos

Saves to:

extracted_features_resnet50/

3️⃣ Train Model (Multi-GPU)
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port 29501 \
  run_training.py


Checkpoints saved in:

checkpoints/

4️⃣ Evaluate Model
python run_evaluation.py \
  --resume /path/to/best_checkpoint.ckpt


Metrics:

mAP@tIoU

Recall@k

Results are logged and printed to console.

📊 Evaluation Metrics

mAP@tIoU – Mean Average Precision at temporal IoU thresholds

Recall@k – Top-k retrieval accuracy

These follow standard temporal grounding benchmarks.

🖼 Example Outputs

(Optional — add visual results if available)

Temporal segment predictions overlaid on video timeline

Query-conditioned grounding examples

📚 Acknowledgment

This repository builds upon:

Moment-DETR: End-to-End Video Instance Segmentation with Transformers
Wang et al., CVPR 2021

If you use this implementation, please cite the original paper.

@inproceedings{momentdetr2021,
  title={End-to-End Video Instance Segmentation with Transformers},
  author={Wang, Yuqing and Xu, Zhaoliang and Wang, Xinlong and Li, Chun-Guang and Yao, Yong-Qiang and Li, Yue-Meng and Meng, Gaofeng},
  booktitle={CVPR},
  year={2021}
}
