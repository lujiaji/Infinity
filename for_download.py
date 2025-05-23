# import pandas as pd
# import os, requests
# from concurrent.futures import ThreadPoolExecutor
# import json
# import cv2
# import numpy as np

# df = pd.read_parquet("https://huggingface.co/datasets/UCSC-VLAA/Recap-COCO-30K/resolve/main/new_data.parquet")

# # 固定模板值（来源于 Infinity 的 dynamic_resolution_h_w）
# h_div_w_templates = np.array([
#     1.000, 1.250, 1.333, 1.500, 1.750, 2.000, 2.500, 3.000,
#     0.800, 0.750, 0.667, 0.571, 0.500, 0.400, 0.333
# ])

# def match_h_div_w_template(h, w):
#     ratio = h / w
#     return float(h_div_w_templates[np.argmin(np.abs(h_div_w_templates - ratio))])

# with open("meta_info.jsonl", "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
#         coco_id = int(row["image_id"])
#         img_path = f"val2014/COCO_val2014_{coco_id:012d}.jpg"
#         if not os.path.exists(img_path):
#             continue
#         try:
#             img = cv2.imread(img_path)
#             h, w = img.shape[:2]
#             h_div_w = match_h_div_w_template(h, w)
#         except:
#             continue  # 图像损坏等异常跳过

#         f.write(json.dumps({
#             "image_path": img_path,
#             "text": row["caption"],
#             "long_caption": row["recaption"],
#             "h_div_w": h_div_w
#         }, ensure_ascii=False) + "\n")

from modelscope.hub.snapshot_download import snapshot_download

# 这是官网给出的 model_id
model_id = 'iic/mplug_visual-question-answering_coco_large_en'
# 指定下载到的本地目录
local_dir = '/home/jiaji_lu/.cache/modelscope/mplug_vqa'

# 下载所有文件到 local_dir
snapshot_download(model_id, cache_dir=local_dir)

print("✅ mPLUG 下载完毕，路径：", local_dir)
