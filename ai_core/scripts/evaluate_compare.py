import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from PIL import Image
import cv2

# Cấu hình đường dẫn tuyệt đối cho import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_CORE_PATH = PROJECT_ROOT
sys.path.append(str(AI_CORE_PATH))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.resnet_transfer import ResNetTransfer
from models.efficientnet_transfer import EfficientNetTransfer
from models.densenet_transfer import DenseNetTransfer
from xai.gradcam_runner.gradcam_runner import XAIAnalyzer
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Đọc cấu hình
CONFIG_PATH = AI_CORE_PATH / "config.yml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config['training']['batch_size']
IMG_SIZE = config['training']['image_size']
NUM_CLASSES = config['model']['num_classes']
# Tự động chọn đường dẫn checkpoints ở local hoặc Kaggle
CHECKPOINT_DIR = PROJECT_ROOT / config['paths']['checkpoint_dir']
if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
COMPARISON_DIR = AI_CORE_PATH
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """Tự động tìm và tải các mô hình đã được train xong"""
    models_dict = {}
    
    # 1. ResNet
    resnet_path = CHECKPOINT_DIR / "resnet" / "best_model_resnet.pth"
    if not resnet_path.exists():
        resnet_path = CHECKPOINT_DIR / "best_model_resnet.pth"
    if resnet_path.exists():
        model = ResNetTransfer(num_classes=NUM_CLASSES)
        checkpoint = torch.load(resnet_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        model.to(DEVICE).eval()
        models_dict["ResNet"] = {"model": model, "target_layer": model.resnet.layer4[-1], "history": checkpoint.get("history", None)}

    # 2. DenseNet
    dense_path = CHECKPOINT_DIR / "densenet" / "best_model_densenet.pth"
    if not dense_path.exists():
        dense_path = CHECKPOINT_DIR / "best_model_densenet.pth"
    if dense_path.exists():
        model = DenseNetTransfer(num_classes=NUM_CLASSES)
        checkpoint = torch.load(dense_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        model.to(DEVICE).eval()
        models_dict["DenseNet"] = {"model": model, "target_layer": model.densenet.features.denseblock4, "history": checkpoint.get("history", None)}
        
    # 3. EfficientNet
    eff_path = CHECKPOINT_DIR / "efficientnet" / "best_model_efficientnet.pth"
    if not eff_path.exists():
        eff_path = CHECKPOINT_DIR / "best_model_efficientnet.pth"
    if eff_path.exists():
        model = EfficientNetTransfer(num_classes=NUM_CLASSES)
        checkpoint = torch.load(eff_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        model.to(DEVICE).eval()
        models_dict["EfficientNet"] = {"model": model, "target_layer": model.efficientnet.features[-1], "history": checkpoint.get("history", None)}
        
    return models_dict

def run_inference(models_dict, test_loader):
    """Quét tập Test và ghi lại kết quả"""
    results = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in models_dict.keys()}
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Đang Inference"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels_np = labels.cpu().numpy()
            
            for name, m_info in models_dict.items():
                outputs = m_info["model"](images)
                probs = F.softmax(outputs, dim=1)[:, 1] # Lấy xác suất của class 1 (REAL)
                preds = torch.argmax(outputs, dim=1)
                
                results[name]["y_true"].extend(labels_np)
                results[name]["y_pred"].extend(preds.cpu().numpy())
                results[name]["y_prob"].extend(probs.cpu().numpy())
                
    return results

def plot_roc_curves(results):
    plt.figure(figsize=(10, 8))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Tỷ lệ Dương tính Giả)', fontsize=12)
    plt.ylabel('True Positive Rate (Tỷ lệ Dương tính Thật)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(COMPARISON_DIR / "roc_auc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(results):
    n_models = len(results)
    if n_models == 0: return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1: axes = [axes]
    
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 16})
        ax.set_title(f'{name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.xaxis.set_ticklabels(['FAKE', 'REAL'], fontsize=12)
        ax.yaxis.set_ticklabels(['FAKE', 'REAL'], fontsize=12)
        
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_barchart(results):
    models = list(results.keys())
    if not models: return
    
    accs, precs, recs, f1s = [], [], [], []
    for name in models:
        y_true, y_pred = results[name]["y_true"], results[name]["y_pred"]
        accs.append(accuracy_score(y_true, y_pred))
        precs.append(precision_score(y_true, y_pred))
        recs.append(recall_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))
        
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Màu sắc tương đồng với ảnh mẫu
    rects1 = ax.bar(x - width*1.5, accs, width, label='Độ chính xác\n(Accuracy)', color='#00a2e8')
    rects2 = ax.bar(x - width*0.5, precs, width, label='Tỷ lệ trúng\n(Precision)', color='#d9534f')
    rects3 = ax.bar(x + width*0.5, recs, width, label='Độ nhạy\n(Recall)', color='#93c47d')
    rects4 = ax.bar(x + width*1.5, f1s, width, label='F1-Score\n', color='#7e7ac8') # Thêm \n để ngang hàng với các label khác
    
    # ax.set_ylabel('Scores', fontsize=12) # Có thể bỏ ylabel cho giống ảnh
    ax.set_title('Kết quả đánh giá hiệu năng của các mô hình', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14, fontweight='bold')
    
    # Legend ở dưới cùng (bottom)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=12, frameon=False)
    
    # Căn chỉnh trục Y để thấy rõ sự khác biệt (như ảnh mẫu 0.92 -> 1.0)
    all_scores = accs + precs + recs + f1s
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    # Trục Y bắt đầu thấp hơn min_score một chút, và kết thúc ở 1.0 (hoặc cao hơn tí để nhét chữ)
    y_min = max(0.0, float(min_score) - 0.05)
    y_max = min(1.0, float(max_score) + 0.05)
    ax.set_ylim([y_min, y_max])
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, rotation=0)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    plt.grid(axis='y', alpha=0.3)
    plt.subplots_adjust(bottom=0.25) # Chừa khoảng trống cho legend
    plt.savefig(COMPARISON_DIR / "metrics_barchart.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curves(models_dict):
    valid_models = {name: info for name, info in models_dict.items() if info.get("history") is not None}
    if not valid_models: return
    
    # Mapping colors
    color_map = {
        "ResNet": '#1f77b4',       # Xanh nước biển
        "DenseNet": '#d62728',     # Đỏ
        "EfficientNet": '#2ca02c'   # Xanh lá cây
    }
    
    for name, info in valid_models.items():
        hist = info["history"]
        if "train_loss" not in hist or "val_loss" not in hist: continue
        
        color = color_map.get(name, '#1f77b4')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        epochs = range(1, len(hist["train_loss"]) + 1)
        
        # Loss Curve
        axes[0].plot(epochs, hist["train_loss"], linestyle='--', color=color, label=f'{name} (Train)')
        axes[0].plot(epochs, hist["val_loss"], linewidth=2, linestyle='-', color=color, label=f'{name} (Val)')
        axes[0].set_title(f'Learning Curve (Loss) - {name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epochs', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy Curve
        axes[1].plot(epochs, hist["train_acc"], linestyle='--', color=color, label=f'{name} (Train)')
        axes[1].plot(epochs, hist["val_acc"], linewidth=2, linestyle='-', color=color, label=f'{name} (Val)')
        axes[1].set_title(f'Learning Curve (Accuracy) - {name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epochs', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Determine filename
        if "resnet" in name.lower():
            filename = "resnet_learning_curve.png"
        elif "efficientnet" in name.lower():
            filename = "efficientnet_learning_curve.png"
        elif "densenet" in name.lower():
            filename = "densenet_learning_curve.png"
        else:
            filename = f"{name.lower()}_learning_curve.png"
            
        plt.savefig(COMPARISON_DIR / filename, dpi=300, bbox_inches='tight')
        plt.close()

def generate_gradcam_grid(models_dict, test_loader):
    if not models_dict: return
    # Tìm đúng 2 ảnh REAL và 2 ảnh FAKE để phân tích cân bằng
    real_images, fake_images = [], []
    real_labels, fake_labels = [], []
    
    for images, labels in test_loader:
        for i in range(len(labels)):
            if labels[i].item() == 1 and len(real_images) < 2:
                real_images.append(images[i:i+1])
                real_labels.append(labels[i:i+1])
            elif labels[i].item() == 0 and len(fake_images) < 2:
                fake_images.append(images[i:i+1])
                fake_labels.append(labels[i:i+1])
                
            if len(real_images) == 2 and len(fake_images) == 2:
                break
        if len(real_images) == 2 and len(fake_images) == 2:
            break
            
    if not real_images and not fake_images: return
    
    selected_images = torch.cat(real_images + fake_images).to(DEVICE)
    selected_labels = torch.cat(real_labels + fake_labels).to(DEVICE)
    num_images = len(selected_images)
    
    fig, axes = plt.subplots(num_images, len(models_dict) + 1, figsize=(4 * (len(models_dict) + 1), 4 * num_images))
    if num_images == 1: axes = [axes]
    
    for i in range(num_images):
        img_tensor = selected_images[i:i+1]
        label = "REAL" if selected_labels[i].item() == 1 else "FAKE"
        
        # Denormalize image for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
        display_img = img_tensor[0] * std + mean
        display_img = display_img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        
        # Original Image Column
        ax = axes[i][0]
        ax.imshow(display_img)
        ax.set_title(f"Ảnh Gốc\nNhãn: {label}", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        col = 1
        for name, m_info in models_dict.items():
            ax = axes[i][col]
            
            # Predict
            output = m_info["model"](img_tensor)
            pred_class = torch.argmax(output).item()
            pred_label = "REAL" if pred_class == 1 else "FAKE"
            
            # Grad-CAM
            xai = XAIAnalyzer(model=m_info["model"], target_layer=m_info["target_layer"], device=DEVICE)
            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = xai.cam(input_tensor=img_tensor, targets=targets)[0, :]
            
            # Overlay
            visualization = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)
            
            ax.imshow(visualization)
            color = "green" if pred_class == selected_labels[i].item() else "red"
            ax.set_title(f"{name}\nDự đoán: {pred_label}", color=color, fontsize=14, fontweight='bold')
            ax.axis('off')
            col += 1
            
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "gradcam_comparison_grid.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*60)
    print("BẮT ĐẦU CHẠY ĐÁNH GIÁ & SO SÁNH CÁC MÔ HÌNH")
    print("="*60)
    
    models_dict = load_models()
    if not models_dict:
        print("Không tìm thấy file weights (.pth) nào trong thư mục ai_core/checkpoints!")
        print("Vui lòng copy file trọng số của ít nhất 1 mô hình vào đúng thư mục trước khi chạy.")
        return
        
    print(f"Đã tìm thấy và tải thành công các mô hình: {list(models_dict.keys())}")
    
    print("\nĐang nạp Dữ liệu Test...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # TRÊN KAGGLE: Sửa TEST_DIR thành đường dẫn tới thư mục test của bạn (ví dụ: "/kaggle/input/cifake/test")
    TEST_DIR = PROJECT_ROOT / config['paths']['test_dir']
    
    if not TEST_DIR.exists():
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại {TEST_DIR}")
        print("Vui lòng sửa lại biến TEST_DIR trong code trỏ đúng tới dataset trên Kaggle của bạn.")
        return
        
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("\nĐang chạy quá trình nhận diện (Inference) trên tập Test...")
    results = run_inference(models_dict, test_loader)
    
    print("\nĐang vẽ các biểu đồ phân tích (ROC, Confusion Matrix, Bar Chart)...")
    plot_roc_curves(results)
    plot_confusion_matrices(results)
    plot_metrics_barchart(results)
    plot_learning_curves(models_dict)
    
    print("\nĐang sinh ảnh Lưới bản đồ nhiệt Grad-CAM...")
    generate_gradcam_grid(models_dict, test_loader)
    
    print("\nHOÀN TẤT TỐT ĐẸP! Toàn bộ biểu đồ đã được lưu tại:")
    print(COMPARISON_DIR)

if __name__ == "__main__":
    main()