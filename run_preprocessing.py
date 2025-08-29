"""
Moment-DETR Baseline: Step 1 - Preprocessing (Server-Ready)
"""
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

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

def main():
    print("[INFO] Starting preprocessing for Moment-DETR.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    for split, csv_path in SPLIT_MAPPING.items():
        if not Path(csv_path).exists():
            print(f"[ERROR] Cannot find CSV for '{split}': {csv_path}"); continue
        print(f"--> Processing split: {split} from {csv_path}")
        df = pd.read_csv(csv_path)
        grouped_data, video_col, text_col = defaultdict(list), 'video_id' if 'video_id' in df.columns else 'video', 'text' if 'text' in df.columns else 'query'
        for _, row in df.iterrows():
            video_id, query = str(row[video_col]), row[text_col]
            start_time, end_time = row['start_frame'] / cfg.FRAME_RATE, row['end_frame'] / cfg.FRAME_RATE
            grouped_data[(video_id, query)].append([start_time, end_time])
        durations = {str(r[video_col]): r['end_frame'] / cfg.FRAME_RATE for _, r in df.groupby(video_col)['end_frame'].max().reset_index().iterrows()}
        output_path = OUTPUT_DIR / f"{split}.jsonl"
        with open(output_path, "w") as f:
            for (video_id, query), segments in grouped_data.items():
                duration = durations.get(video_id, segments[-1][1])
                f.write(json.dumps({"video": video_id, "duration": duration, "query": query, "timestamps": segments}) + "\n")
        print(f"    Saved {len(grouped_data)} records to {output_path}")
    print("\n[SUCCESS] Preprocessing complete.")

if __name__ == "__main__": main()

