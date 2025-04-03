import csv
import os
from distutils.log import info
import pickle
from matplotlib.pyplot import contour, show
import numpy as np
import cv2 as cv
from tqdm import tqdm

# 标记图片地址
source_img_dir = 'final_result/yolo/extend/labelframes'

# 中间产出文件地址根目录，最终文件保存地址根目录
middle_res_root = 'middle_result'
final_result = 'final_result'

# via标注文件地址
origin_csv_path = 'final_result/yolo/extend/via_drink_phone_write.csv'

# 中间文件产出
out_action_list = f'{middle_res_root}/ava_action_list_v2.1.pbtxt'
out_labelmap_path = f'{middle_res_root}/labelmap.txt'

# 最终文件产出地址
# yolo_labels = f'{final_result}/yolo/train/labels'
yolo_labels = 'final_result/yolo/extend/labels'

def get_label_map(origin_csv_path, out_action_list, out_labelmap_path):
    classes_list = 0
    classes_content = ""
    labelmap_strings = ""
    # 提取出csv中的第9行的行为下标
    with open(origin_csv_path, 'r', encoding='utf-8') as csvfile:
        count = 0
        content = csv.reader(csvfile)
        for line in content:
            if count == 8:
                classes_list = line
                break
            count += 1
    # 截取种类字典段落
    st = 0
    ed = 0
    for i in range(len(classes_list)):
        if classes_list[i].startswith('options'):
            st = i
        if classes_list[i].startswith('default_option_id'):
            ed = i
    for i in range(st, ed):
        if i == st:
            classes_content = classes_content + classes_list[i][len('options:'):] + ','
        else:
            classes_content = classes_content + classes_list[i] + ','
    classes_dict = eval(classes_content)[0]
    # 写入labelmap.txt文件
    with open(out_action_list, 'w', encoding='utf-8') as f:  # 写入action_list文件
        for v, k in classes_dict.items():
            labelmap_strings = labelmap_strings + "label {{\n  name: \"{}\"\n  label_id: {}\n  label_type: PERSON_MOVEMENT\n}}\n".format(
                k, int(v) + 1)
        f.write(labelmap_strings)
    labelmap_strings = ""
    with open(out_labelmap_path, 'w', encoding='utf-8') as f:  # 写入label_map文件
        for v, k in classes_dict.items():
            labelmap_strings = labelmap_strings + "{}: {}\n".format(int(v) + 1, k)
        f.write(labelmap_strings)


def via2yolo(origin_csv_path, dataset_percent=1):
    information_array = [[], [], []]
    # 读取输入csv文件的位置信息段落
    with open(origin_csv_path, 'r', encoding='utf-8') as csvfile:
        count = 0
        content = csv.reader(csvfile)
        for line in content:
            # print(line)
            if count >= 10:
                frame_image_name = eval(line[1])[0]  # str
                # print(line[-2])
                location_info = eval(line[4])[1:]  # list
                action_list = list(eval(line[5]).values())[0].split(',')
                # print(action_list)
                action_list = [int(x) for x in action_list]  # list
                information_array[0].append(frame_image_name)
                information_array[1].append(location_info)
                information_array[2].append(action_list)
            count += 1
    # 将：对应帧图片名字、物体位置信息、动作种类信息汇总为一个信息数组
    information_array = np.array(information_array, dtype=object).transpose()
    print(information_array[0]) #['10_0_000001.jpg' list([392, 151, 79, 162]) list([4])]
    for info in tqdm(information_array):
        vide_name = info[0]
        x = info[1][0]
        y = info[1][1]
        w = info[1][2]
        h = info[1][3]
        img_path = os.path.join(source_img_dir, vide_name)
        img = cv.imread(img_path)
        print('----'.join([source_img_dir,vide_name]))
        img_h, img_w = img.shape[:2]
        label_id = info[2][0]
        yolo_x = (x + w / 2) / img_w
        yolo_y = (y + h / 2) / img_h
        yolo_w = w / img_w
        yolo_h = h / img_h
        label_name = vide_name.replace('.jpg', '.txt')
        label_path = os.path.join(yolo_labels, label_name)
        if not os.path.exists(label_path):
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(' '.join([str(label_id), str(yolo_x), str(yolo_y), str(yolo_w), str(yolo_h)]))
                f.write('\n')
        else:
            with open(label_path, 'a', encoding='utf-8', newline='\n') as f:
                f.write(' '.join([str(label_id), str(yolo_x), str(yolo_y), str(yolo_w), str(yolo_h)]))
                f.write('\n')



    # information_array = np.array(information_array)
    # -----------------------------------------------------------------------------------------------
    # num_train = int(dataset_percent * len(information_array))
    # train_info_array = information_array[:num_train]
    # valid_info_array = information_array[num_train:]

# 从via文件获取动作类别，并生成ava_action_list_v2.1.pbtxt和labelmap.txt
# get_label_map(origin_csv_path, out_action_list, out_labelmap_path)

via2yolo(origin_csv_path, 1)