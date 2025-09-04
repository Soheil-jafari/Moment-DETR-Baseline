import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class MomentDETRDataset(Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.split = split
        self.cfg = cfg
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        ann_path = os.path.join(cfg.ann_path, f"{split}.jsonl")
        self.annotations = [json.loads(line) for line in open(ann_path, 'r')]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        feature_path = os.path.join(self.cfg.feature_path, f"{ann['video']}.npz")
        features = np.load(feature_path)['features'].astype(np.float32)

        if features.shape[0] > self.cfg.max_v_len:
            indices = np.linspace(0, features.shape[0] - 1, self.cfg.max_v_len).astype(int)
            features = features[indices]

        # Tokenize query (this part is unchanged and correct)
        word_ids = self.tokenizer(
            ann['query'],
            add_special_tokens=True,
            max_length=self.cfg.max_q_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        # --- START OF CORRECTED GROUND TRUTH LOGIC ---

        # Get timestamps [[start,end]] in seconds
        timestamps = torch.tensor(ann['timestamps'], dtype=torch.float32)

        # Normalize timestamps to the range [0, 1] based on the video's total duration
        spans_start_end = timestamps / float(ann['duration'])

        # Directly convert the normalized [start, end] to [center, width]
        spans_center = (spans_start_end[:, 0] + spans_start_end[:, 1]) / 2
        spans_width = (spans_start_end[:, 1] - spans_start_end[:, 0])

        # Final [center, width] tensor
        spans = torch.stack([spans_center, spans_width], dim=-1)

        # --- END OF CORRECTED GROUND TRUTH LOGIC ---

        return {'video_feats': torch.from_numpy(features), 'query': word_ids['input_ids'].squeeze(0),
                'query_mask': word_ids['attention_mask'].squeeze(0), 'spans': spans, 'duration': ann['duration'],
                'video_id': ann['video'], 'query_str': ann['query']}

def collate_fn(batch):
    v_feats = torch.zeros(len(batch), max(x['video_feats'].shape[0] for x in batch), batch[0]['video_feats'].shape[1])
    v_mask = torch.zeros(len(batch), v_feats.shape[1], dtype=torch.bool)
    for i, item in enumerate(batch):
        len_v = item['video_feats'].shape[0]; v_feats[i, :len_v] = item['video_feats']; v_mask[i, :len_v] = True
    queries = torch.stack([x['query'] for x in batch])
    query_masks = torch.stack([x['query_mask'] for x in batch])
    targets = [{'spans': item['spans'], 'labels': torch.zeros(len(item['spans']), dtype=torch.long)} for item in batch]
    return {'video_feats': v_feats, 'video_mask': v_mask, 'query': queries, 'query_mask': query_masks, 'targets': targets,
            'meta': {'video_id': [x['video_id'] for x in batch], 'duration': [x['duration'] for x in batch], 'query': [x['query_str'] for x in batch]}}

