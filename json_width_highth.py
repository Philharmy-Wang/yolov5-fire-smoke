import pandas as pd
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['font.family'] = 'sans-serif'  # 设置字体样式
plt.rcParams['figure.figsize'] = (10.0, 10.0)  # 设置字体大小

# 读取数据
ann_json_path = "/home/gb/detectron2/datasets/gun_detection/annotations/gun_train.json"
with open(ann_json_path) as f:
    ann = json.load(f)

# 创建{1: 'multi_signs', 2: 'window_shielding', 3: 'non_traffic_sign'}
# 创建{'multi_signs': 0, 'window_shielding': 0, 'non_traffic_sign': 0}
categorys_dic = dict([(i['id'], i['name']) for i in ann['categories']])
categorys_num = dict([i['name'], 0] for i in ann['categories'])

# 统计每个类别的数量
for i in ann['annotations']:
    categorys_num[categorys_dic[i['category_id']]] += 1

# 统计bbox的w、h、wh
bbox_w = []
bbox_h = []
bbox_wh = []
for i in ann['annotations']:
    bbox_w.append(round(i['bbox'][2], 2))
    bbox_h.append(round(i['bbox'][3], 2))
    wh = round(i['bbox'][2] / i['bbox'][3], 0)
    if (wh < 1):
        wh = round(i['bbox'][3] / i['bbox'][2], 0)
    bbox_wh.append(wh)

# 统计所有的宽高比
bbox_wh_unique = set(bbox_wh)  # set挑选出不重复的元素，即挑选出有多少种比例的anchors
# print(bbox_wh_unique)
bbox_count_unique = [bbox_wh.count(i) for i in bbox_wh_unique]  # 统计宽高比数量
# print(bbox_count_unique)

# 画图
wh_dataframe = pd.DataFrame(bbox_count_unique, index=bbox_wh_unique, columns=['宽高比数量'])
wh_dataframe.plot(kind='bar', color="#55aacc")
plt.show
