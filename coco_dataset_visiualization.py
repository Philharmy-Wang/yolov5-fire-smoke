import json
from pycocotools.coco import COCO
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

coco = COCO('/home/gb/yolov5/VOCdevkit/VOC2007/gun_all_train.json')

# coco = COCO('./yourpath/coco_train.json')
# dict_keys = coco.imgs.keys()# 将所有imgs的id提取出来，并创建了元组，元组包含了一个由imgs的id组成的列表
# list_keys = list(dict_keys)
# num = len(list_keys)

#分析图片数量、标记框数量和类别数量
img_num_list = coco.get_img_ids()
ann_num_list = coco.get_ann_ids()
cat_num_list = coco.get_cat_ids()
img_num = len(img_num_list)
ann_num = len(ann_num_list)
cat_num = len(cat_num_list)
print('图片数量： {} \n标记框数量： {} \n类别数量: {}'.format(img_num, ann_num, cat_num))

# imgs = coco.loadAnns(ann_num_list[0])
# cats = coco.loadCats(cat_num_list[0])

#分析图片高宽分布
size = []
for i in range(img_num):
    h = coco.load_imgs(img_num_list[i])[0]['height']
    w = coco.load_imgs(img_num_list[i])[0]['width']
    size.append((h, w))
#print(len(size))
h_w_cls = set(size)
h_w_num = len(h_w_cls)
for i in h_w_cls:
    print('高宽为{}的图片数量为：{}'.format(i,size.count(i)))
print('宽高比例的类别数：', h_w_num)

#分析数据集中是否有重复图片
file_name_list = []
for i in range(img_num):
    file_name = coco.load_imgs(img_num_list[i])[0]['file_name']
    file_name_list.append(file_name)
img_notrepeat_list = list(set(file_name_list))
img_notrepeat_num = len(img_notrepeat_list)
print('图片数量： {}\n统计不重复图片：{}'.format(img_num,img_notrepeat_num))

#分析每个类别的数量
category_id_list = []
for i in range(ann_num):
    category_id = coco.loadAnns(ann_num_list[i])[0]['category_id']
    category_id_list.append(category_id)
cat_1_num = category_id_list.count(1)
#print(ategory_id_list)
cat_2_num = category_id_list.count(2)
cat_3_num = category_id_list.count(3)
print('图片中类别1的数量有：{}\n图片中类别2的数量有：{}\n图片中类别3的数量有:{}'.format(cat_1_num, cat_2_num, cat_3_num))

#可视化类别比例
plt.figure(figsize=(6, 9))
labels = ['muti_signs', 'window_shielding', 'non_traffic_sign']
sizes = [cat_1_num, cat_2_num, cat_3_num]
colors = ['red','yellowgreen','lightskyblue']
explode = (0.05,0,0)

patches,l_text,p_text = plt.pie(sizes,explode=explode,labels=labels,colors=colors,
                                labeldistance = 1.1,autopct = '%3.1f%%',shadow = False,
                                startangle = 90,pctdistance = 0.6)
#改变文本的大小
#方法是把每一个text遍历。调用set_size方法设置它的属性
for t in l_text:
    t.set_size=(30)
for t in p_text:
    t.set_size=(20)
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.legend()
plt.show()

