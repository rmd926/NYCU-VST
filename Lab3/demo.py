import os
import cv2
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import argparse

'''
Command Line:
python demo.py -iv easy_9.mp4 -ov easy_9_output.mp4 -w yolov7.pt -c 0.35 -i 0.5 -m 64 -d 60
python demo.py -iv hard_9.mp4 -ov hard_9_output.mp4 -w yolov7.pt -c 0.35 -i 0.5 -m 160 -d 120
'''

# 設置設備
device = select_device('0')  # 使用GPU 0

# 載入 YOLOv7 模型
def load_yolov7_model(weights='yolov7.pt', img_size=640):
    model = attempt_load(weights, map_location=device)  # 加載 YOLOv7 權重
    model.eval()
    return model, img_size

def parse_args():
    parser = argparse.ArgumentParser(description="Using YOLOv7 as Detection Model add Hungarian Algo to do object tracking")
    
    parser.add_argument('-iv', '--input_video', type=str, default='easy_9.mp4', help="Path to the input video")
    parser.add_argument('-ov', '--output_video', type=str, required='easy_9_output.mp4', help="Path to the output video")
    parser.add_argument('-w', '--weights', type=str, default='yolov7.pt', help="Path to the YOLOv7 weights file")
    parser.add_argument('-s', '--img_size', type=int, default=640, help="Image size for YOLOv7 inference")
    parser.add_argument('-c', '--conf_thresh', type=float, default=0.35, help="Confidence threshold for detection")
    parser.add_argument('-i', '--iou_thresh', type=float, default=0.5, help="IoU threshold for non-max suppression")
    parser.add_argument('-m', '--max_disappeared', type=int, default=64, help="Max frames an object can disappear before being removed")
    parser.add_argument('-d', '--distance_thresh', type=int, default=60, help="Distance threshold for matching objects")
    
    return parser.parse_args()


def compute_giou(box1, box2):
    """
    計算 Generalized IoU (gIoU) 的值
    Args:
        box1: 第一個框，格式為 [x1, y1, x2, y2]
        box2: 第二個框，格式為 [x1, y1, x2, y2]
    Returns:
        gIoU 值 (範圍: [-1, 1])
    """
    # 計算交集框
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 計算框的面積
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    # 計算 IoU
    iou = intersection / union if union > 0 else 0

    # 計算最小包圍框 (Enclosing Box)
    enclosing_x1 = min(box1[0], box2[0])
    enclosing_y1 = min(box1[1], box2[1])
    enclosing_x2 = max(box1[2], box2[2])
    enclosing_y2 = max(box1[3], box2[3])
    enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)

    # 避免除以零
    if enclosing_area == 0:
        return iou

    # 計算 gIoU
    giou = iou - ((enclosing_area - union) / enclosing_area)

    # 確保返回值合法
    if np.isnan(giou) or np.isinf(giou):
        return 0
    return giou


# 計算中心點距離(Euclidean)
def compute_center_distance(box1, box2):
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
    return distance

# 判斷框是否在邊界附近
def is_near_boundary(box, width, height, margin=20):
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= (width - margin) or y2 >= (height - margin)

# 匹配框，使用 GIOU進行配對
def match_boxes(previous_boxes, current_boxes, iou_threshold=0.10): #threshold若設太高反而容易ID switch
    cost_matrix = np.zeros((len(previous_boxes), len(current_boxes)))

    for i, prev_box in enumerate(previous_boxes):
        for j, curr_box in enumerate(current_boxes):
            iou_cost = 1 - compute_giou(prev_box, curr_box) # cost 介於[0,2]
            cost_matrix[i, j] = iou_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_pairs = []
    unmatched_current = set(range(len(current_boxes)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < (1 - iou_threshold):
            matched_pairs.append((r, c))
            unmatched_current.discard(c)

    return matched_pairs, list(unmatched_current)

def update_trajectories(obj_id, center, trajectories, max_length=60):
    if obj_id in trajectories:
        if len(trajectories[obj_id]) > 0:
            last_point = trajectories[obj_id][-1]
            if np.linalg.norm(np.array(center) - np.array(last_point)) > max_length:
                trajectories[obj_id].clear()
                trajectories[obj_id].append(center)
            else:
                trajectories[obj_id].append(center)
                if len(trajectories[obj_id]) > max_length:
                    trajectories[obj_id].popleft()
        else:
            trajectories[obj_id].append(center)
    else:
        trajectories[obj_id] = deque([center], maxlen=max_length)

def draw_trajectories(frame, trajectories, color_dict):
    for obj_id, points in trajectories.items():
        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i - 1]))
            pt2 = tuple(map(int, points[i]))
            cv2.line(frame, pt1, pt2, color_dict[obj_id], 2)

# 處理影片
def process_video(input_video, output_video, model, img_size, conf_thresh, iou_thresh, max_disappeared,distance_thresh):
    cap = cv2.VideoCapture(input_video)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    dynamic_output_video = f"{base_name}_output.mp4"


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (original_width, original_height))

    previous_boxes = []
    previous_ids = []
    next_id = 0
    color_dict = defaultdict(lambda: (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
    object_memory = {}
    last_positions = {}
    speeds = {}
    trajectories = defaultdict(deque)
    frame_count = 0

    scale_x = original_width / img_size
    scale_y = original_height / img_size

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        img = cv2.resize(frame, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=[0], agnostic=False)

        current_boxes = []
        confidences = []
        if len(pred) > 0 and pred[0] is not None:
            det = pred[0].cpu().numpy()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                current_boxes.append([x1, y1, x2, y2])
                confidences.append(conf)

        matched_pairs, unmatched_current = match_boxes(previous_boxes, current_boxes,iou_threshold=0.10)

        current_ids = [-1] * len(current_boxes)
        frame_used_ids = set()

        for prev_idx, curr_idx in matched_pairs:
            obj_id = previous_ids[prev_idx]
            if obj_id not in frame_used_ids:
                current_ids[curr_idx] = obj_id
                frame_used_ids.add(obj_id)
                center_x = (current_boxes[curr_idx][0] + current_boxes[curr_idx][2]) / 2
                center_y = (current_boxes[curr_idx][1] + current_boxes[curr_idx][3]) / 2
                update_trajectories(obj_id, (center_x, center_y), trajectories)
                object_memory[obj_id] = {'last_seen': frame_count, 'box': current_boxes[curr_idx]}
                if frame_count % 10 == 0:
                    if obj_id in last_positions:
                        speed_x = center_x - last_positions[obj_id][0]
                        speed_y = center_y - last_positions[obj_id][1]
                        speeds[obj_id] = (speed_x, speed_y)
                    last_positions[obj_id] = (center_x, center_y)
            else:
                current_ids[curr_idx] = next_id
                next_id += 1
        for idx in unmatched_current:
            found_memory = False
            for obj_id, mem_info in list(object_memory.items()):
                if frame_count - mem_info['last_seen'] < max_disappeared and not is_near_boundary(mem_info['box'], original_width, original_height):
                    if compute_giou(mem_info['box'], current_boxes[idx]) > 0.5:  # 使用 GIoU 匹配
                        current_ids[idx] = obj_id
                        center_x = (current_boxes[idx][0] + current_boxes[idx][2]) / 2
                        center_y = (current_boxes[idx][1] + current_boxes[idx][3]) / 2
                        update_trajectories(obj_id, (center_x, center_y), trajectories)
                        object_memory[obj_id] = {'last_seen': frame_count, 'box': current_boxes[idx]}
                        found_memory = True
                        break

            if not found_memory:
                # 基於距離匹配最近的記憶框
                min_distance = float('inf')
                closest_id = None
                for obj_id, mem_info in object_memory.items():
                    distance = compute_center_distance(mem_info['box'], current_boxes[idx])
                    if distance < min_distance and obj_id not in current_ids:
                        min_distance = distance
                        closest_id = obj_id

                if min_distance < distance_thresh and closest_id not in current_ids:
                    # 使用距離最近的 ID
                    current_ids[idx] = closest_id
                    center_x = (current_boxes[idx][0] + current_boxes[idx][2]) / 2
                    center_y = (current_boxes[idx][1] + current_boxes[idx][3]) / 2
                    update_trajectories(closest_id, (center_x, center_y), trajectories)
                    object_memory[closest_id] = {'last_seen': frame_count, 'box': current_boxes[idx]}
                else:
                    # 分配新 ID
                    current_ids[idx] = next_id
                    next_id += 1
                    center_x = (current_boxes[idx][0] + current_boxes[idx][2]) / 2
                    center_y = (current_boxes[idx][1] + current_boxes[idx][3]) / 2
                    update_trajectories(next_id, (center_x, center_y), trajectories)
                    object_memory[next_id] = {'last_seen': frame_count, 'box': current_boxes[idx]}

        # 清理過期物件和對應的軌跡
        expired_ids = [k for k, v in object_memory.items() if frame_count - v['last_seen'] > max_disappeared]
        object_memory = {k: v for k, v in object_memory.items() if frame_count - v['last_seen'] <= max_disappeared}
        for obj_id in expired_ids:
            if obj_id in trajectories:
                del trajectories[obj_id]

        current_max_id = max(current_ids) + 1 if current_ids else 0
        cv2.putText(frame, f'count: {current_max_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        dynamic_output_video = f"{base_name}_{current_max_id}_output.mp4"

        draw_trajectories(frame, trajectories, color_dict)

        for box, obj_id, conf in zip(current_boxes, current_ids, confidences):
            color = color_dict[obj_id]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID {obj_id},Conf {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if obj_id in speeds:
                speed_vector = speeds[obj_id]
                speed_magnitude = np.linalg.norm(speed_vector)
                speed_text = f"Speed: {speed_magnitude/10:.2f}px/frame"
                cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            print(f"Frame #: {frame_count}, Tracker ID: {obj_id}, Class: person, BBox Coords (xmin, ymin, xmax, ymax): ({x1}, {y1}, {x2}, {y2})")

            for i in range(1, len(trajectories[obj_id])):
                pt1 = tuple(map(int, trajectories[obj_id][i - 1]))
                pt2 = tuple(map(int, trajectories[obj_id][i]))
                cv2.line(frame, pt1, pt2, color, 2)

        out.write(frame)
        previous_boxes = current_boxes
        previous_ids = current_ids

    cap.release()
    out.release()
    print(f"Video processing completed.")

if __name__ == "__main__":
    args = parse_args()

    model, img_size = load_yolov7_model(weights=args.weights, img_size=args.img_size)
    process_video(
        input_video=args.input_video,
        output_video=args.output_video,
        model=model,
        img_size=args.img_size,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        max_disappeared=args.max_disappeared,
        distance_thresh=args.distance_thresh
    )


# if __name__ == "__main__":
#     input_video = 'easy_9.mp4'
#     output_video = 'easy_output_with_trajectory.mp4'

#     model, img_size = load_yolov7_model(weights='yolov7.pt')
#     process_video(input_video, output_video, model, img_size, conf_thresh=0.35, iou_thresh=0.5, max_disappeared=64) 
