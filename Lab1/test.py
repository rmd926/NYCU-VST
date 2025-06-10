# import torch
# from torchvision import transforms,datasets
# from PIL import Image
# import os
# import pandas as pd
# from model import ClassificationModel
# from tqdm import tqdm

# # 設置數據集路徑與批次大小
# data_dir = r'C:\Users\user\Desktop\hw1_313553024'
# test_dir = os.path.join(data_dir, 'test')  # 測試資料夾路徑
# batch_size = 32

# # 檢查是否有可用的 GPU，如果沒有則使用 CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Using device: {device}')

# # 定義數據增強與轉換（與訓練過程一致）
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # 加載模型和權重，將模型移到 GPU
# model = ClassificationModel(num_classes=100).to(device)
# model.load_state_dict(torch.load('w_313553024.pth'))
# model.eval()

# # 加載訓練資料集以獲取類別名稱
# train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
# classes = train_dataset.classes  # 取得每個類別的名稱列表

# # 預測並保存結果
# predictions = []

# # 遍歷測試資料夾中的所有圖片
# with torch.no_grad():
#     # 使用 tqdm 來顯示進度條
#     with tqdm(os.listdir(test_dir), desc='Testing', unit="image") as ttest:
#         for image_name in ttest:
#             # 加載圖片並應用數據增強與轉換
#             image_path = os.path.join(test_dir, image_name)
#             image = Image.open(image_path).convert('RGB')  # 確保圖片為RGB格式
#             input_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch 維度

#             # 模型預測
#             outputs = model(input_tensor)

#             # 取得 top-5 預測結果，並將它們轉換為類別名稱
#             _, predicted_top5 = torch.topk(outputs, 5, dim=1)
#             predicted_top5 = predicted_top5.cpu().numpy().tolist()[0]  # 轉換為 list
#             predicted_classes = [classes[i] for i in predicted_top5]   # 將數字轉換為類別名稱

#             # 保存預測結果
#             predictions.append([image_name] + predicted_classes)

# # 保存預測到 CSV 文件
# df = pd.DataFrame(predictions, columns=["file_name", "pred1", "pred2", "pred3", "pred4", "pred5"])
# df.to_csv(f'pred_313553024.csv', index=False)
import torch
from torchvision import transforms, datasets
from PIL import Image
import os
import pandas as pd
from model import ClassificationModel
from tqdm import tqdm

# Set the dataset path and batch size
data_dir = r'C:\Users\user\Desktop\hw1_313553024'
test_dir = os.path.join(data_dir, 'test')  # Path to the test folder
batch_size = 32

# Check if GPU is available; if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define data augmentation and transformations (consistent with the training process)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the model and weights, move the model to GPU
model = ClassificationModel(num_classes=100).to(device)
model.load_state_dict(torch.load('w_313553024.pth'))
model.eval()

# Load the training dataset to get the class names
train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
classes = train_dataset.classes  # Get the list of class names

# Predict and save results
predictions = []

# Iterate through all images in the test folder
with torch.no_grad():
    # Use tqdm to display the progress bar
    with tqdm(os.listdir(test_dir), desc='Testing', unit="image") as ttest:
        for image_name in ttest:
            # Load the image and apply data augmentation and transformation
            image_path = os.path.join(test_dir, image_name)
            image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
            input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

            # Model prediction
            outputs = model(input_tensor)

            # Get top-5 predictions and convert them to class names
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            predicted_top5 = predicted_top5.cpu().numpy().tolist()[0]  # Convert to list
            predicted_classes = [classes[i] for i in predicted_top5]   # Convert indices to class names

            # Save the prediction results
            predictions.append([image_name] + predicted_classes)

# Save predictions to a CSV file
df = pd.DataFrame(predictions, columns=["file_name", "pred1", "pred2", "pred3", "pred4", "pred5"])
df.to_csv(f'pred_313553024.csv', index=False)
