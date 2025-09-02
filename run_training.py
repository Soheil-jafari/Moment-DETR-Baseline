"""
Moment-DETR Baseline: Step 3 - Training
"""
import os, argparse, datetime, torch, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from moment_detr_module.configs import Config
from moment_detr_module.modeling import MomentDETR
from moment_detr_module.dataset import MomentDETRDataset, collate_fn
from moment_detr_module.engine import train_one_epoch, evaluate
from moment_detr_module.utils import setup_seed, get_logger
from tqdm import tqdm

def main(args):
    if args.dist: dist.init_process_group(backend="nccl"); torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank); cfg = Config(); setup_seed(cfg.seed)
    run_name = f"moment_detr_{cfg.dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(args.ckpt_dir, run_name)
    if args.local_rank == 0: os.makedirs(checkpoint_dir, exist_ok=True)
    logger = get_logger(os.path.join(checkpoint_dir, "train.log")) if args.local_rank == 0 else None
    if args.local_rank == 0: logger.info(f"Saving checkpoints to: {checkpoint_dir}"); logger.info(
        f"Config: {vars(cfg)}")
    model = MomentDETR(cfg).to(device)
    if args.dist: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]}, {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": cfg.lr_backbone}]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop)
    train_dataset, val_dataset = MomentDETRDataset(cfg, 'train'), MomentDETRDataset(cfg, 'val')
    train_sampler, val_sampler = (DistributedSampler(train_dataset) if args.dist else None), (DistributedSampler(val_dataset, shuffle=False) if args.dist else None)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, sampler=val_sampler)
    print("\n" + "=" * 40)
    print("!!! RUNNING OVERFITTING SANITY CHECK !!!")
    print("=" * 40 + "\n")

    # Get just one batch from the training loader
    single_batch = next(iter(train_loader))

    # Overwrite the loaders to always return this single batch
    train_loader = [single_batch]
    val_loader = [single_batch]
    # --- END OF SANITY CHECK CODE ---

    if args.local_rank == 0: logger.info("Start training")
    if args.local_rank == 0: logger.info("Start training")
    best_metric = -1.0
    for epoch in tqdm(range(cfg.epochs), desc="Training Epochs"):
        if args.dist: train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, device, epoch, cfg.clip_max_norm, logger, args.local_rank)
        lr_scheduler.step()
        if args.local_rank == 0:
            # Check if running in distributed mode to save the correct model state
            state_dict = model.module.state_dict() if args.dist else model.state_dict()

            # Save the epoch checkpoint
            torch.save(
                {'model': state_dict, 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),
                 'epoch': epoch}, os.path.join(checkpoint_dir, f"epoch_{epoch}.ckpt"))

            # Run evaluation
            if (epoch + 1) % args.eval_every == 0 or epoch == cfg.epochs - 1:
                eval_stats = evaluate(model, val_loader, device)
                logger.info(f"Validation - Epoch {epoch}: {eval_stats}")
                current_metric = eval_stats.get('R1@0.5', -1)

                # Save the best model checkpoint
                if current_metric > best_metric:
                    best_metric = current_metric
                    torch.save({'model': state_dict}, os.path.join(checkpoint_dir, "best_checkpoint.ckpt"))
                    logger.info(f"Saved new best model with R1@0.5: {best_metric:.4f}")

    if args.local_rank == 0: logger.info("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8); parser.add_argument("--ckpt_dir", type=str, default="checkpoints"); parser.add_argument("--eval_every", type=int, default=1); parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(); args.dist = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1; main(args)

