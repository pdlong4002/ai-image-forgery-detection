import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
import random

from utils.metrics import calculate_metrics

# Import models
from models.resnet_transfer import ResNetTransfer
from models.efficientnet_transfer import EfficientNetTransfer
from models.densenet_transfer import DenseNetTransfer


def setup_dataloaders(config, project_root, device):
    """Thiết lập Transforms, Datasets và DataLoaders."""
    img_size = config['training']['image_size']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = project_root / config['paths']['train_dir']
    test_dir = project_root / config['paths']['test_dir']
    
    # Khởi tạo dataset gốc
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    full_val_dataset = datasets.ImageFolder(root=train_dir, transform=test_transforms)
    
    print(f"Class mapping: {full_train_dataset.class_to_idx}")
    
    # Tách 10% cho validation từ tập train theo phân tầng (stratified) thuần Python để cân bằng FAKE/REAL
    targets = full_train_dataset.targets
    
    class_0_idx = [i for i, label in enumerate(targets) if label == 0]
    class_1_idx = [i for i, label in enumerate(targets) if label == 1]
    
    # Trộn ngẫu nhiên (cố định seed 42)
    rng = random.Random(42)
    rng.shuffle(class_0_idx)
    rng.shuffle(class_1_idx)
    
    val_size_0 = int(0.1 * len(class_0_idx))
    val_size_1 = int(0.1 * len(class_1_idx))
    
    val_idx = class_0_idx[:val_size_0] + class_1_idx[:val_size_1]
    train_idx = class_0_idx[val_size_0:] + class_1_idx[val_size_1:]
    
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_val_dataset, val_idx)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    
    # Chỉ bật pin_memory nếu đang dùng GPU (cuda) để tránh UserWarning trên CPU
    use_pin_memory = True if device.type == 'cuda' else False
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    print(f"Train: {len(train_dataset)} ảnh | Validation: {len(val_dataset)} ảnh | Test: {len(test_dataset)} ảnh")
    
    return train_loader, val_loader, test_loader


def build_model(config, device):
    """Khởi tạo mô hình dựa trên cấu hình."""
    model_name = config['model']['name'].lower()
    num_classes = config['model']['num_classes']
    fine_tune = config['model'].get('fine_tune', False)
    
    if model_name == "resnet":
        model = ResNetTransfer(num_classes=num_classes, fine_tune=fine_tune)
    elif model_name == "efficientnet":
        model = EfficientNetTransfer(num_classes=num_classes, fine_tune=fine_tune)
    elif model_name == "densenet":
        model = DenseNetTransfer(num_classes=num_classes, fine_tune=fine_tune)
    else:
        model = ResNetTransfer(num_classes=num_classes, fine_tune=fine_tune)
        
    model = model.to(device)
    return model, model_name


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Huấn luyện mô hình trong 1 Epoch."""
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []
    device_type = device.type
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        
    avg_loss = train_loss / len(dataloader)
    metrics = calculate_metrics(train_labels, train_preds)
    return avg_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    """Đánh giá mô hình trên tập Validation trong 1 Epoch."""
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    device_type = device.type
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    avg_loss = val_loss / len(dataloader)
    metrics = calculate_metrics(val_labels, val_preds)
    return avg_loss, metrics


def main():
    """Hàm chính điều phối toàn bộ quá trình huấn luyện."""
    PROJECT_ROOT = Path(__file__).resolve().parent
    with open(PROJECT_ROOT / "config.yml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. Khởi tạo dữ liệu
    train_loader, val_loader, test_loader = setup_dataloaders(config, PROJECT_ROOT, device)
    
    # 2. Khởi tạo mô hình
    model, model_name = build_model(config, device)

    # 3. Khởi tạo các thành phần huấn luyện (Loss, Optimizer, Scheduler, Scaler)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    device_type = device.type
    scaler = torch.amp.GradScaler(device=device_type, enabled=(device_type == 'cuda'))

    # 4. Thiết lập thư mục Checkpoint và khôi phục trạng thái
    epochs = config['training']['epochs']
    patience = config['training'].get('patience', 5)
    
    checkpoint_dir = PROJECT_ROOT / config['paths']['checkpoint_dir']
    model_checkpoint_dir = checkpoint_dir / model_name
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    LAST_SAVE_PATH = model_checkpoint_dir / f"last_model_{model_name}.pth"
    BEST_SAVE_PATH = model_checkpoint_dir / f"best_model_{model_name}.pth"

    start_epoch = 0
    best_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if os.path.exists(LAST_SAVE_PATH):
        print(f"Khôi phục checkpoint từ {LAST_SAVE_PATH}")
        checkpoint = torch.load(LAST_SAVE_PATH, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        history = checkpoint.get("history", history)
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        print(f"Tiếp tục từ Epoch {start_epoch}")

    print("\n" + "="*70)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print("="*70)

    # 5. Vòng lặp huấn luyện chính
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        # Train & Evaluate
        avg_train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        avg_val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        # In kết quả
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train Acc : {train_metrics['accuracy']:.4f} | Val Acc : {val_metrics['accuracy']:.4f}")

        # Cập nhật Scheduler
        scheduler.step(val_metrics['accuracy'])

        # Lưu lịch sử
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_metrics['accuracy'])
        history["val_acc"].append(val_metrics['accuracy'])

        # Kiểm tra mô hình tốt nhất
        is_best = val_metrics['accuracy'] > best_acc
        if is_best:
            best_acc = val_metrics['accuracy']
            epochs_no_improve = 0
            print("Best model updated!")
        else:
            epochs_no_improve += 1

        # Lưu Checkpoints
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "history": history,
            "epochs_no_improve": epochs_no_improve
        }
        
        torch.save(checkpoint, LAST_SAVE_PATH)
        if is_best:
            torch.save(checkpoint, BEST_SAVE_PATH)

        # Dừng sớm
        if epochs_no_improve >= patience:
            print("Early Stopping!")
            break


if __name__ == '__main__':
    main()