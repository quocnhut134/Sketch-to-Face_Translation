import os
import torch
import json
from tqdm import tqdm
from torch.cuda.amp import GradScaler

class Trainer:
    def __init__(self, strategy, train_loader, val_loader, train_cfg):
        self.strategy = strategy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_cfg = train_cfg
        self.strategy_cfg = strategy.strategy_cfg 
        self.device = train_cfg.device

        self.scaler = GradScaler(enabled=(self.device == 'cuda'))
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        
        output_dir = self.strategy_cfg["output_dir"]
        self.best_model_path = os.path.join(output_dir, "best_model")
        self.latest_model_path = os.path.join(output_dir, "latest_model")
        self.log_file_path = os.path.join(output_dir, "training_logs.json")
        os.makedirs(output_dir, exist_ok=True)
        
        self.training_logs = []

    def train_epoch(self, epoch):
        self.strategy.trainable_model.train() 
        epoch_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training")
        
        for step, batch in enumerate(progress_bar):            
            loss = self.strategy.train_step(batch, self.scaler)
            epoch_loss += loss
            
            if (step + 1) % self.train_cfg.accumulation_steps == 0:
                self.scaler.step(self.strategy.optimizer)
                self.scaler.update()
                self.strategy.optimizer.zero_grad()
            
            progress_bar.set_postfix(Loss=f"{loss:.4f}")
            
        avg_train_loss = epoch_loss / len(self.train_loader)
        print(f"\nEpoch {epoch}, Avg Train Loss: {avg_train_loss:.4f}")
        return avg_train_loss

    def validate_epoch(self, epoch):
        self.strategy.trainable_model.eval() 
        val_loss = 0
        val_progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation")
        
        for batch in val_progress_bar:
            loss = self.strategy.val_step(batch)
            val_loss += loss
            val_progress_bar.set_postfix(Val_Loss=f"{val_loss / len(self.val_loader):.4f}")
            
        avg_val_loss = val_loss / len(self.val_loader)
        print(f"\nEpoch {epoch}, Avg Val Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def train(self):
        for epoch in range(self.train_cfg.num_epochs):
            avg_train_loss = self.train_epoch(epoch)
            avg_val_loss = self.validate_epoch(epoch)
            
            self.strategy.scheduler.step()
            
            epoch_log = {
                'epoch': epoch + 1, 
                'avg_train_loss': avg_train_loss, 
                'avg_val_loss': avg_val_loss
            }
            self.training_logs.append(epoch_log)
            try:
                with open(self.log_file_path, 'w') as f:
                    json.dump(self.training_logs, f, indent=4)
            except Exception as e:
                print(e)

            # Early stopping & Model saving
            if avg_val_loss < self.best_eval_loss:
                self.best_eval_loss = avg_val_loss
                self.patience_counter = 0
                self.strategy.save_model(self.best_model_path)
                print(f"Saved best model at: {self.best_model_path}")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter} / {self.train_cfg.patience}")
                if self.patience_counter >= self.train_cfg.patience:
                    print(f"Early stopping after {epoch + 1} epochs.")
                    break
        
        self.strategy.save_model(self.latest_model_path)
        print(f"Saved final model at: {self.latest_model_path}")
        print(f"Model saved (Val Loss: {self.best_eval_loss:.4f}) at: {self.best_model_path}")