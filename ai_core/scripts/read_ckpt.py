import torch
import sys

ckpt_path = 'D:/VS_CODE_OwO/Projects/deep-learning/dl-ai-image-forgery-detection/ai_core/checkpoints/efficientnet/best_model_efficientnet.pth'
try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', 0) + 1
    best_acc = ckpt.get('best_acc', 0)
    history = ckpt.get('history', {})
    print("\n" + "="*40)
    print("KẾT QUẢ CỦA BEST MODEL EFFICIENTNET")
    print("="*40)
    print(f"- Số epoch đã train: {epoch}")
    print(f"- Accuracy cao nhất (best_acc): {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("\n- Lịch sử Loss & Accuracy:")
    if history:
        for k, v in history.items():
            if v:
                print(f"  + {k}: {[round(x, 4) for x in v]}")
            else:
                print(f"  + {k}: Chưa có dữ liệu")
    print("="*40 + "\n")
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
