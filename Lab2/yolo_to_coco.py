import os
import json

yolo_train_dir = './datasets/train_labels'  
train_img_dir = './datasets/train2017'  
train_output_json = './datasets/annotations/train_coco.json'

yolo_val_dir = './datasets/val_labels'  
val_img_dir = './datasets/val2017'  
val_output_json = './datasets/annotations/val_coco.json'

def yolo_to_coco(yolo_label, img_width, img_height):
    x_center, y_center, w, h = map(float, yolo_label[1:])
    x_min = (x_center - w / 2) * img_width
    x_max = (x_center + w / 2) * img_width 
    y_min = (y_center - h / 2) * img_height
    y_max = (y_center + h / 2) * img_height 
    width = w * img_width
    height = h * img_height
    return [x_min, y_min, width, height]#cuz coco only needs xmin and ymin, no needed to show xmax ymax

def convert_yolo_to_coco(yolo_dir, output_json, img_dir):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "car"}],  
    }
    ann_id = 1
    img_id = 1
    for img_file in os.listdir(img_dir):
        if img_file.endswith(".jpg"):  
            img_path = os.path.join(img_dir, img_file)
            img_width, img_height = 1024, 576  
            coco_data["images"].append({
                "id": img_id,
                "file_name": img_file,
                "width": img_width,
                "height": img_height
            })
            label_file = os.path.join(yolo_dir, f"{os.path.splitext(img_file)[0]}.txt")
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        yolo_label = line.strip().split()
                        bbox = yolo_to_coco(yolo_label, img_width, img_height)
                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(yolo_label[0]),  # 類別ID
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],  # width * height
                            "iscrowd": 0
                        })
                        ann_id += 1
            img_id += 1

    with open(output_json, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

convert_yolo_to_coco(yolo_train_dir, train_output_json, train_img_dir)
convert_yolo_to_coco(yolo_val_dir, val_output_json, val_img_dir)

print("Having Generated train_coco.json and val_coco.json")
