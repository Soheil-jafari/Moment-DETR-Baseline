# Moment-DETR-Baseline
Moment-DETR Baseline Experiment for Paper
This folder contains the official implementation of Moment-DETR, adapted to run on your Cholec80 dataset for the temporal grounding baseline.

This is a multi-step process. You must run the scripts in the specified order.

Workflow Overview
run_preprocessing.py: Converts your cholec80_*.csv files into the JSONL format required by the training script.

run_feature_extraction.py: Processes all video frames through a ResNet50 model to generate feature vectors (.npz files), which are the actual input to Moment-DETR. This is computationally intensive and will take time.

run_training.py: Trains the Moment-DETR model using the preprocessed annotations and extracted features.

run_evaluation.py: Runs the trained model on the test set to predict [start, end] timestamps and calculates the mAP@tIoU and Recall@k metrics for your paper.

Step 1: Setup the Environment (Do This Only Once)
Create a new conda environment for this baseline. It has different dependencies than the X-CLIP one.

# Navigate to this directory
cd moment_detr_baseline

# Create and activate the conda environment
conda create -n detr python=3.9 -y
conda activate detr

# Install PyTorch with CUDA support (adjust for your server)
pip install torch torchvision --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install the rest of the dependencies
pip install -r requirements.txt

Step 2: Run the Pipeline Scripts
Execute the following scripts in sequence.

Script 1: Preprocess Annotations
This will create train.jsonl, val.jsonl, and test.jsonl files in the preprocessed_data directory.

python run_preprocessing.py

Script 2: Extract Visual Features
This is the most time-consuming step. It will process all frames in your extracted_frames directory and save a .npz feature file for each video. The script will automatically skip videos that have already been processed, so you can stop and resume it.

python run_feature_extraction.py

Script 3: Train the Model
This will start the main training process and save checkpoints in the checkpoints directory.

# This command will train the model.
# The number of GPUs is detected automatically.
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 run_training.py

Note: Adjust nproc_per_node to the number of GPUs you want to use (e.g., 2, 4, 8).

Script 4: Evaluate the Model
After training, use this script to generate the final results for your paper. Make sure to point it to the best checkpoint saved during training.

# Replace '.../best_checkpoint.ckpt' with the actual path to your saved model
python run_evaluation.py --resume /path/to/your/checkpoints/moment_detr_cholec80_.../best_checkpoint.ckpt

The final metrics will be printed to the console and saved in a text file.
