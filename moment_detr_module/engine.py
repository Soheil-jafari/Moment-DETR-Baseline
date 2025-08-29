import torch, numpy as np
from .utils import RunningMeter, get_iou

def train_one_epoch(model, loader, optimizer, device, epoch, max_norm, logger, rank):
    model.train(); loss_meter = RunningMeter()
    for i, data in enumerate(loader):
        video_feats, video_mask, query, query_mask = data['video_feats'].to(device), data['video_mask'].to(device), data['query'].to(device), data['query_mask'].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in data['targets']]
        loss = sum(model(video_feats, video_mask, query, query_mask, targets)['loss_dict'].values())
        optimizer.zero_grad(); loss.backward()
        if max_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step(); loss_meter.update(loss.item())
        if rank == 0 and (i + 1) % 50 == 0: logger.info(f"Epoch {epoch} | Step {i+1}/{len(loader)} | Loss: {loss_meter.avg:.4f}")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = {'R1@0.5': [], 'R1@0.7': [], 'mAP@0.5': [], 'mAP': []}
    for data in loader:
        video_feats, video_mask, query, query_mask = data['video_feats'].to(device), data['video_mask'].to(device), data['query'].to(device), data['query_mask'].to(device)
        pred_spans = model(video_feats, video_mask, query, query_mask)['pred_spans']
        for i in range(len(pred_spans)):
            pred, gt, duration = pred_spans[i].cpu().numpy() * data['meta']['duration'][i], data['targets'][i]['spans'].cpu().numpy() * data['meta']['duration'][i], data['meta']['duration'][i]
            iou = get_iou(pred[0:1], gt)
            metrics['R1@0.5'].append(iou.max() >= 0.5); metrics['R1@0.7'].append(iou.max() >= 0.7)
            ap_05 = calculate_ap(pred, gt, 0.5); ap_075 = calculate_ap(pred, gt, 0.75); ap_095 = calculate_ap(pred, gt, 0.95)
            metrics['mAP@0.5'].append(ap_05); metrics['mAP'].append(np.mean([ap_05, ap_075, ap_095]))
    for key in metrics: metrics[key] = np.mean(metrics[key])
    return metrics

def calculate_ap(pred, gt, iou_thresh):
    ious = get_iou(pred, gt); tp = np.sum(ious >= iou_thresh, axis=1) > 0
    if tp.sum() == 0: return 0.0
    return (np.cumsum(tp) / (np.arange(len(tp)) + 1)).mean()

