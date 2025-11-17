import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SketchToPhotoDataset(Dataset):
    def __init__(self, hed_dir, photo_dir, img_size=512, max_samples=None, grayscale_condition=False):
        self.hed_dir = hed_dir
        self.photo_dir = photo_dir
        self.hed_files = sorted(os.listdir(hed_dir))
        self.photo_files = sorted(os.listdir(photo_dir))
        
        assert len(self.hed_files) == len(self.photo_files), "The number of photos and sketches does not fit."

        if max_samples and len(self.hed_files) > max_samples:
            indices = random.sample(range(len(self.hed_files)), max_samples)
            self.hed_files = [self.hed_files[i] for i in indices]
            self.photo_files = [self.photo_files[i] for i in indices]
            
        self.condition_mode = "L" if grayscale_condition else "RGB"
        
        self.condition_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        self.target_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
    
    def __len__(self):
        return len(self.hed_files)

    def __getitem__(self, idx):
        hed_path = os.path.join(self.hed_dir, self.hed_files[idx])
        photo_path = os.path.join(self.photo_dir, self.photo_files[idx])
        
        hed_image = Image.open(hed_path).convert(self.condition_mode)
        photo_image = Image.open(photo_path).convert("RGB")
        
        hed_tensor = self.condition_transform(hed_image)
        photo_tensor = self.target_transform(photo_image)
        
        return {"condition_image": hed_tensor, "target_image": photo_tensor}
        