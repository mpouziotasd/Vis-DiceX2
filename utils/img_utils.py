import cv2
import numpy as np

color_ranges = {
    "red": (0, 0, 255), 
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255)
}

def draw_fps(frame, fps):
    cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
    
def draw_oriented_bbox(frame, bboxes, dice_nums): # Used primarily for OBB detection. Project is oriented around BB not OBB.
    for bbox, dice_num in zip(bboxes, dice_nums):
        points = np.array(bbox, dtype=np.int32).reshape(4, 2)
        dice_num = int(dice_num.item()) + 1

        cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)
    return frame


def draw_data(frame, bboxes, dice_nums, dice_analytics):
    for bbox, dice_num, dice_color in zip(bboxes, dice_nums, dice_analytics['dice_colors']):
        x_min, y_min, x_max, y_max = map(int, bbox)
        dice_num = int(dice_num.item()) + 1
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
        
        text_position = (x_min, y_min - 10)
        text = f"{dice_color} {dice_num}"

        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    per_clr_sum = dice_analytics['per-clr-sum']
    y_offset = 30 
    x_start = frame.shape[1] - 200
    
    for color, value in sorted(per_clr_sum.items()):
        text = f"{color.capitalize()} Sum: {value}"
        _, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)[0]

        cv2.putText(frame, text, (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)  # Outline
        cv2.putText(frame, text, (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color_ranges[color], thickness=2, lineType=cv2.LINE_AA)  # Main text
        
        y_offset += text_height + 10
    
    return frame


