import torch
import torch.nn as nn
from torchvision import models

class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=2, fine_tune=False):
        super().__init__()
        
        # 1. Tải mô hình ResNet50 đã được huấn luyện trước trên ImageNet
        # Lưu ý: Sử dụng weights=DEFAULT là tiêu chuẩn mới của PyTorch (thay thế cho pretrained=True)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 2. Đóng băng trọng số
        # Nếu fine_tune=False, chỉ huấn luyện đầu ra (Classifier), giữ nguyên bộ trích xuất đặc trưng của ResNet
        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # 3. Thay thế lớp Fully Connected (FC) cuối cùng của ResNet
        in_features = self.resnet.fc.in_features # Với ResNet50, giá trị này là 2048
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),               
            nn.Linear(in_features, 512),  
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)    
        )
        
    def forward(self, x):
        return self.resnet(x)


# Mã kiểm thử
if __name__ == '__main__':
    model = ResNetTransfer(num_classes=2, fine_tune=False)
    
    fake_images = torch.randn(8, 3, 224, 224)
    print(f"\nInput shape:  {fake_images.shape} (Batch=8, Channels=3, W=224, H=224)")

    output = model(fake_images)
    print(f"Output shape: {output.shape} (Batch=8, Classes=2)")