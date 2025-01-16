import numpy as np


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
