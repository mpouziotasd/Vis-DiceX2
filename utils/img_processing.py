import numpy as np
import cv2

def get_dice_analytics(frame, bboxes, dice_nums, sam_model=None):
    per_clr_sum = {}
    dice_analytics = {"dice_colors": []}
    height, width = frame.shape[:2]

    for bbox, dice_num in zip(bboxes, dice_nums):
        x1, y1, x2, y2 = map(int, bbox)
        
        x1, y1, x2, y2 = max(0, x1-5), max(0, y1-5), min(width, x2+5), min(height, y2+5)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            print(f"Empty ROI for bounding box: {x1}, {y1}, {x2}, {y2}")
            continue

        avg_color = np.mean(roi, axis=(0, 1))
        avg_color_bgr = avg_color.astype(int)
        
        if avg_color_bgr[1] > avg_color_bgr[2] and avg_color_bgr[1] > avg_color_bgr[0]:
            color = "green"
        else:
            color = "white"

        dice_analytics['dice_colors'].append(color)

        per_clr_sum[color] = per_clr_sum.get(color, 0) + int(dice_num.item()) + 1

    dice_analytics['per-clr-sum'] = per_clr_sum
    return dice_analytics

def process_and_draw_data(frame, bboxes, dice_nums):
    per_clr_sum = {}
    dice_analytics = {"dice_colors": []}
    height, width = frame.shape[:2]

    frame = frame.astype(np.float32)

    y_offset = 30
    x_start = frame.shape[1] - 200

    for bbox, dice_num in zip(bboxes, dice_nums):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1, x2, y2 = max(0, x1-5), max(0, y1-5), min(width, x2+5), min(height, y2+5)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        subsample_ratio = 4
        subsampled_roi = roi[::subsample_ratio, ::subsample_ratio]
        avg_color = np.mean(subsampled_roi, axis=(0, 1))
        avg_color_bgr = avg_color.astype(int)

        if avg_color_bgr[1] > avg_color_bgr[2] and avg_color_bgr[1] > avg_color_bgr[0]:
            color = "green"
        else:
            color = "white"

        dice_analytics['dice_colors'].append(color)

        per_clr_sum[color] = per_clr_sum.get(color, 0) + int(dice_num.item()) + 1

        dice_num = int(dice_num.item()) + 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        text_position = (x1, y1 - 10)
        text = f"{color} {dice_num}"
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for color, value in sorted(per_clr_sum.items()):
        summary_text = f"{color.capitalize()} Sum: {value}"
        _, text_height = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)[0]

        cv2.putText(frame, summary_text, (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(frame, summary_text, (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0) if color == "green" else (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)  # Main text

        y_offset += text_height + 10

    dice_analytics['per-clr-sum'] = per_clr_sum
    return frame, dice_analytics