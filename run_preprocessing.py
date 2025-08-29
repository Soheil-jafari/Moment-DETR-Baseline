"""
Moment-DETR Baseline: Step 1 - Preprocessing (Server-Ready & Corrected)

This script is now updated to read the frame-by-frame annotation format
(frame_path, text_query, relevance_label) from your CSV files and convert it
into the contiguous [start, end] time segments required by Moment-DETR.
"""
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm

# --- Configuration (Paths from your project_config.py) ---
class ServerConfig:
    UNIFIED_MEDICAL_VIDEOS_DIR = "/home/240331715/data/unified_medical_videos"
    TRAIN_CSV = f"{UNIFIED_MEDICAL_VIDEOS_DIR}/final_triplets/cholec80_train_triplets.csv"
    VAL_CSV = f"{UNIFIED_MEDICAL_VIDEOS_DIR}/final_triplets/cholec80_val_triplets.csv"
    TEST_CSV = f"{UNIFIED_MEDICAL_VIDEOS_DIR}/final_triplets/cholec80_test_triplets.csv"
    FRAME_RATE = 30

cfg = ServerConfig()
OUTPUT_DIR = Path("./preprocessed_data")
SPLIT_MAPPING = {"train": cfg.TRAIN_CSV, "val": cfg.VAL_CSV, "test": cfg.TEST_CSV}

def extract_info_from_path(path_str):
    """Extracts video_id and frame_idx from a path like '.../CHOLEC80__video01/frame_0000123.jpg'"""
    video_id_match = re.search(r'(CHOLEC80__video\d+)', path_str)
    frame_idx_match = re.search(r'frame_(\d+)\.jpg', path_str)
    video_id = video_id_match.group(1) if video_id_match else None
    frame_idx = int(frame_idx_match.group(1)) if frame_idx_match else -1
    return video_id, frame_idx

def merge_consecutive_frames_to_segments(frames):
    """Merges a sorted list of frame indices into [start, end] segments."""
    if not frames:
        return []
    segments = []
    start_frame = frames[0]
    end_frame = frames[0]
    for i in range(1, len(frames)):
        if frames[i] == end_frame + 1:
            end_frame = frames[i]
        else:
            segments.append([start_frame, end_frame + 1]) # +1 for exclusive end
            start_frame = frames[i]
            end_frame = frames[i]
    segments.append([start_frame, end_frame + 1])
    return segments

def main():
    print("[INFO] Starting preprocessing for Moment-DETR (Updated for frame-level CSVs).")
    OUTPUT_DIR.mkdir(exist_ok=True)

    for split, csv_path in SPLIT_MAPPING.items():
        if not Path(csv_path).exists():
            print(f"[ERROR] Cannot find CSV for '{split}': {csv_path}"); continue
        
        print(f"--> Processing split: {split} from {csv_path}")
        df = pd.read_csv(csv_path)

        # Extract video_id and frame_idx from the 'frame_path' column
        print("    Extracting info from frame paths...")
        path_info = df['frame_path'].apply(extract_info_from_path)
        df['video_id'] = [info[0] for info in path_info]
        df['frame_idx'] = [info[1] for info in path_info]
        
        # Filter for positive labels and group by video and query
        positive_frames = df[df['relevance_label'] == 1]
        
        # Group positive frames by video and query
        grouped = positive_frames.groupby(['video_id', 'text_query'])['frame_idx'].apply(list).reset_index()

        output_path = OUTPUT_DIR / f"{split}.jsonl"
        records_written = 0
        
        print(f"    Merging frames into segments for {len(grouped)} (video, query) pairs...")
        with open(output_path, "w") as f:
            for _, row in tqdm(grouped.iterrows(), total=len(grouped)):
                video_id, query, frame_indices = row['video_id'], row['text_query'], sorted(row['frame_idx'])
                
                # Merge consecutive frames into segments
                segments_in_frames = merge_consecutive_frames_to_segments(frame_indices)
                
                # Convert frame segments to time segments
                segments_in_seconds = [[s / cfg.FRAME_RATE, e / cfg.FRAME_RATE] for s, e in segments_in_frames]
                
                # Find max frame to estimate duration
                max_frame = df[df['video_id'] == video_id]['frame_idx'].max()
                duration = (max_frame + 1) / cfg.FRAME_RATE

                record = {
                    "video": video_id,
                    "duration": duration,
                    "query": query,
                    "timestamps": segments_in_seconds
                }
                f.write(json.dumps(record) + "\n")
                records_written += 1
                
        print(f"    Saved {records_written} records to {output_path}")

    print("\n[SUCCESS] Preprocessing complete. Data is now in the correct format for Moment-DETR.")

if __name__ == "__main__":
    main()

