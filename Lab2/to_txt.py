# import torch
# import cv2
# from pathlib import Path
# from yolox.exp import get_exp
# from yolox.data.data_augment import ValTransform
# from yolox.utils import postprocess

# # 設置模型配置和模型加載
# exp = get_exp('exps/example/custom/yolox_s.py', 'yolox-s')  # 替換為實際的 exp 文件和模型名稱
# model = exp.get_model()
# ckpt = torch.load('YOLOX_outputs/yolox_s/epoch_161_ckpt.pth', map_location='cpu')
# model.load_state_dict(ckpt['model'])
# model.eval()
# model.cuda()

# # 設置圖像處理
# preproc = ValTransform(legacy=False)

# # 圖像目錄
# image_dir = Path('datasets/val2017')
# image_paths = list(image_dir.glob('*.jpg'))

# # 讀取和預處理圖像
# imgs = []
# for path in image_paths:
#     img = cv2.imread(str(path))
#     img, _ = preproc(img, None, exp.test_size)
#     imgs.append(torch.from_numpy(img).unsqueeze(0))
# batch = torch.cat(imgs, 0).cuda()

# # 推論
# with torch.no_grad():
#     outputs = model(batch)
#     # 設定置信度閾值為0.7，NMS閾值為0.65
#     outputs = postprocess(outputs, exp.num_classes, 0.7, 0.65)

# # 顯示結果
# for i, output in enumerate(outputs):
#     if output is not None:
#         output = output.cpu()
#         bboxes = output[:, 0:4]
#         scores = output[:, 4] * output[:, 5]
#         for j in range(bboxes.size(0)):
#             bbox = bboxes[j]
#             score = scores[j]
#             print(f"Image: {image_paths[i].name}, Confidence: {score:.2f}, "
#                   f"BBox: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
#epoch_296_ckpt

import os
import torch
import cv2
from pathlib import Path
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

# result_dir = Path('Result')
# result_dir.mkdir(parents=True, exist_ok=True) 
current_dir = Path(__file__).resolve()  # 獲取當前腳本的絕對路徑
parent_dir = current_dir.parents[2]     # 移動到上兩層目錄

# 在上兩層目錄中創建 'Result' 資料夾
result_dir = parent_dir / 'Result'
result_dir.mkdir(parents=True, exist_ok=True)

exp = get_exp('exps/example/custom/yolox_s.py', 'yolox-s') 
model = exp.get_model()
ckpt = torch.load('313553024_ckpt.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])  # 確保模型權重被加載

model.eval()
model.cuda()
preproc = ValTransform(legacy=False)

image_dir = Path('datasets/test_1')
image_paths = list(image_dir.glob('*.jpg'))

# 讀取和預處理圖像
imgs = []
original_sizes = []
for path in image_paths:
    img = cv2.imread(str(path))
    original_sizes.append(img.shape[:2])  # 儲存原始圖像尺寸 (height, width)
    img, _ = preproc(img, None, exp.test_size)  # 這裡 exp.test_size 是 1024x576
    imgs.append(torch.from_numpy(img).unsqueeze(0))
batch = torch.cat(imgs, 0).cuda()

# inference
with torch.no_grad():
    outputs = model(batch)
    # threshold: conf = 0.7 nms = 0.65
    outputs = postprocess(outputs, exp.num_classes, 0.7, 0.65)

# 為每張圖片生成獨立的txt文件，並保存結果
for i, output in enumerate(outputs):
    image_name = image_paths[i].stem 
    txt_file_path = result_dir / f"{image_name}.txt"
    
    with open(txt_file_path, "w") as f:
        if output is not None:
            output = output.cpu()
            bboxes = output[:, 0:4]
            scores = output[:, 4] * output[:, 5]
            original_height, original_width = original_sizes[i]
            scale_w = original_width / exp.test_size[1]  # 寬度縮放比例
            scale_h = original_height / exp.test_size[0]  # 高度縮放比例
            for j in range(bboxes.size(0)):
                bbox = bboxes[j]
                score = scores[j]
                # 邊界框從模型輸入比例（1024x576）轉換為原始圖像比例
                bbox[0] *= scale_w  # x1
                bbox[1] *= scale_h  # y1
                bbox[2] *= scale_w  # x2
                bbox[3] *= scale_h  # y2

                bbox = torch.round(bbox).int()
                f.write(f"0 {score:.6f} {bbox[0].item()} {bbox[1].item()} {bbox[2].item()} {bbox[3].item()}\n")
                
            print(f"Complete and save {txt_file_path}")

