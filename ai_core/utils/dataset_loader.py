import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import yaml

class CIFakeDataset(Dataset):
    """
    Tập dữ liệu tùy chỉnh cho nhận diện ảnh ngụy tạo AI (Deepfake).
    Được thiết kế để tải ảnh thủ công từ thư mục và gán nhãn.
    """
    def __init__(self, root_dir, transform=None):
        self.data_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB') 
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    
    config_path = PROJECT_ROOT / "config.yml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        train_dir = PROJECT_ROOT / config["paths"]["train_dir"]
        
        if train_dir.exists():
            train_dataset = CIFakeDataset(root_dir=train_dir)
            
            print(f"Tổng số ảnh Huấn luyện (Train): {len(train_dataset)}")
            print(f"Tên các lớp: {train_dataset.classes}")
            print(f"Bản đồ lớp sang chỉ mục (Index): {train_dataset.class_to_idx}")
            
            img, label = train_dataset[0]
            print(f"\nThông tin ảnh đầu tiên:")
            print(f" - Kích thước ảnh: {img.size}")
            print(f" - Chỉ mục nhãn (Label Index): {label}")
        else:
            print(f" Không tìm thấy thư mục: {train_dir}. Vui lòng tải tập dữ liệu!")
    else:
        print(f" Không tìm thấy file cấu hình tại: {config_path}")