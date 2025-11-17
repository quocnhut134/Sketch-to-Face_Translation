import os
import argparse
import torch
from torch.utils.data import DataLoader

try:
    from configs.config import data_config, train_config, model_config
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from configs.config import data_config, train_config, model_config

from src.data.dataset import SketchToPhotoDataset
from src.training.strategies import get_strategy
from src.training.trainer import Trainer

def main(strategy_name):
    d_cfg = data_config()
    t_cfg = train_config()
    m_cfg = model_config()
    
    is_grayscale = (strategy_name == "strategy_4") 
        
    train_dataset = SketchToPhotoDataset(
        hed_dir=os.path.join(d_cfg.dataset_dir, "train", "sketches"),
        photo_dir=os.path.join(d_cfg.dataset_dir, "train", "photos"),
        img_size=d_cfg.img_size,
        grayscale_condition=is_grayscale
    )
    
    val_dataset = SketchToPhotoDataset(
        hed_dir=os.path.join(d_cfg.dataset_dir, "val", "sketches"),
        photo_dir=os.path.join(d_cfg.dataset_dir, "val", "photos"),
        img_size=d_cfg.img_size,
        grayscale_condition=is_grayscale
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=t_cfg.batch_size, 
        shuffle=True,
        pin_memory=True, 
        num_workers=os.cpu_count() // 2 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=t_cfg.batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=os.cpu_count() // 2
    )
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")

    strategy = get_strategy(strategy_name, t_cfg, m_cfg)

    trainer = Trainer(
        strategy=strategy,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=t_cfg
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Sketch-to-Face model")
    parser.add_argument(
        "--strategy", 
        type=str, 
        required=True, 
        choices=["strategy_1", "strategy_2", "strategy_3", "strategy_4"],
        help="Choose strategy to train"
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    main(args.strategy)