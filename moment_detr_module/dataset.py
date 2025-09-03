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

            # actual number of tokens after downsampling
        num_tokens = features.shape[0]

        # tokenize query
        word_ids = self.tokenizer(
            ann['query'],
            add_special_tokens=True,
            max_length=self.cfg.max_q_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        # normalize timestamps
        timestamps = torch.tensor(ann['timestamps'], dtype=torch.float32)  # [[start,end]] in seconds
        spans_start_end = timestamps / float(ann['duration'])  # -> [0,1], shape [1,2]

        # ---- convert to center/width
        spans_center = (spans_start_end[:, 0] + spans_start_end[:, 1]) / 2
        spans_width = (spans_start_end[:, 1] - spans_start_end[:, 0])

        # ---- SNAP TO TOKEN GRID (CRITICAL)
        # each token is a bin of width 1/num_tokens. Snap x1 down, x2 up, so the GT
        # is representable by the feature sequence your model sees.
        grid = 1.0 / float(max(1, num_tokens))
        x1 = torch.clamp(torch.floor((spans_center - spans_width / 2) / grid) * grid, 0.0, 1.0)
        x2 = torch.clamp(torch.ceil((spans_center + spans_width / 2) / grid) * grid, 0.0, 1.0)

        # ---- ensure at least ONE token wide (use TWO tokens if your GT is very tiny)
        min_tokens = 2  # try 2 if you still get zeros
        min_w = min_tokens * grid

        cur_w = (x2 - x1)
        need = (min_w - cur_w).clamp(min=0.0)

        # expand symmetrically
        x1 = torch.clamp(x1 - need / 2, 0.0, 1.0)
        x2 = torch.clamp(x2 + need / 2, 0.0, 1.0)

        # after clamping, width might still be < min_w (e.g., when x1 hit 0)
        cur_w = (x2 - x1)
        deficit = (min_w - cur_w).clamp(min=0.0)
        x2 = torch.clamp(x2 + deficit, 0.0, 1.0)  # re-extend to the right

        # recompute center/width after snapping
        spans_center = (x1 + x2) / 2
        spans_width = (x2 - x1)

        # final [cx, w]
        spans = torch.stack([spans_center, spans_width], dim=-1)

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

