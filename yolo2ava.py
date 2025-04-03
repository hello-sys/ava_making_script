import os
import pickle

import numpy as np
from tqdm import tqdm

actionDict = ["look down", "look up", "sleep", "phone", "stand", "drink", "raise hand", "chat", "write"]

ava21_file_list = ['ava_action_list_v2.1.pbtxt',
                     'ava_dense_proposals_train.FAIR.recall_93.9.pkl',
                     'ava_dense_proposals_val.FAIR.recall_93.9.pkl',
                     'ava_train_excluded_timestamps_v2.1.csv',
                     'ava_train_v2.1.csv',
                     'ava_val_excluded_timestamps_v2.1.csv',
                     'ava_val_v2.1.csv',
                     'labelmap.txt']

ava22_file_list = ['ava_detection_train_boxes_and_labels_include_negative_v2.2.csv',
                   'ava_detection_val_boxes_and_labels.csv',
                   'ava_action_list_v2.2_for_activitynet_2019.pbtxt',
                   'ava_val_excluded_timestamps_v2.2.csv',
                   'ava_train_v2.2.csv',
                   'ava_val_v2.2.csv']


def init_ava_files(ann_dir_p, ava_file_list):

    for file in ava_file_list:
        with open(os.path.join(ann_dir_p, file), 'w') as f:
            pass
        print(f'{file} created')


def init_labelmap(labelmap_path):
    with open(labelmap_path, 'w') as f:
        for i, act in enumerate(actionDict, start=1):
            f.write(f'{i}: {act}\n')
    print(f'创建labelmap.txt：{actionDict}')


def init_ava_csv( yolo_label_p , ava_csv_p):
    labels = os.listdir(yolo_label_p)

    with open(ava_csv_p, 'w') as f:

        for txt in tqdm(labels, desc=f'加载：{ava_csv_p}'):
            txt_p = os.path.join(yolo_label_p, txt)

            index_index = '_'.join(txt.split('_')[:2])  # 10_1获取video的id
            seconds_index = int(txt.split('_')[2].split('.')[0])  # 获取秒数 1

            with open(txt_p, 'r') as f_txt:
                txt_lines = f_txt.readlines()

            for line in txt_lines:
                line = line.strip()
                info = line.split(' ')
                if len(info) == 0:
                    continue
                if len(info) == 5:
                    info.append(1)

                act_index = int(info[0]) + 1    #ava的动作id从1开始
                x1 = float(info[1]) - float(info[3])/2
                y1 = float(info[2]) - float(info[4])/2
                x2 = float(info[1]) + float(info[3])/2
                y2 = float(info[2]) + float(info[4])/2
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                people_id = int(info[5])

                f.write(f'{index_index},{seconds_index},{x1},{y1},{x2},{y2},{act_index},{people_id}\n')


def init_b_t_b_a_l_i_n(yolo_label_p , ava_csv_p, without_act_id=False):
    labels = os.listdir(yolo_label_p)

    with open(ava_csv_p, 'w') as f:

        for txt in tqdm(labels, desc=f'加载：{ava_csv_p}'):
            txt_p = os.path.join(yolo_label_p, txt)

            index_index = '_'.join(txt.split('_')[:2])  # 10_1获取video的id
            seconds_index = int(txt.split('_')[2].split('.')[0])  # 获取秒数 1

            with open(txt_p, 'r') as f_txt:
                txt_lines = f_txt.readlines()

            for line in txt_lines:
                line = line.strip()
                info = line.split(' ')
                if len(info) == 0:
                    continue
                if len(info) == 5:
                    info.append(1)

                act_index = int(info[0]) + 1    #ava的动作id从1开始
                x1 = float(info[1]) - float(info[3])/2
                y1 = float(info[2]) - float(info[4])/2
                x2 = float(info[1]) + float(info[3])/2
                y2 = float(info[2]) + float(info[4])/2
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                # people_id = int(info[5])
                precision = 1
                if without_act_id:
                    f.write(f'{index_index},{seconds_index},{x1},{y1},{x2},{y2},,{precision}\n')
                else:
                    f.write(f'{index_index},{seconds_index},{x1},{y1},{x2},{y2},{act_index},{precision}\n')


def init_pkl(yolo_label_p , ava_pkl_p):
    labels = os.listdir(yolo_label_p)

    pkl_data = {}
    for txt in tqdm(labels, desc=f'加载：{ava_pkl_p}'):
        txt_p = os.path.join(yolo_label_p, txt)

        index_index = '_'.join(txt.split('_')[:2])  # 10_1获取video的id
        str_seconds_index = "{:04d}".format(int(txt.split('_')[2].split('.')[0]))  # 字符串秒数'0001'

        with open(txt_p, 'r') as f_txt:
            txt_lines = f_txt.readlines()

        pkl_data_pos = []
        for line in txt_lines:
            line = line.strip()
            info = line.split(' ')
            if len(info) == 0:
                continue

            x1 = float(info[1]) - float(info[3]) / 2
            y1 = float(info[2]) - float(info[4]) / 2
            x2 = float(info[1]) + float(info[3]) / 2
            y2 = float(info[2]) + float(info[4]) / 2
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            pkl_data_pos.append([x1, y1, x2, y2, 1.0])  # 最后一个是置信度默认为1
        key = f'{index_index},{str_seconds_index}'

        pkl_data[key] = pkl_data_pos

    pkl_read_data = {}
    for k, v in pkl_data.items():
        pkl_read_data[k] = np.array(v)

    with open(ava_pkl_p, 'wb') as f:
        pickle.dump(pkl_read_data, f)


def init_action_list_21( ava_action_list_p ,actionDick):

    labelmap_strings = ''
    with open(ava_action_list_p, 'w',encoding='utf-8') as f:  # 写入action_list文件
        for index, act in tqdm(enumerate(actionDick, start=1), desc='写入action_list文件'):
            labelmap_strings = labelmap_strings + "label {{\n  name: \"{}\"\n  label_id: {}\n  label_type: PERSON_MOVEMENT\n}}\n".format(
                act, index)
        f.write(labelmap_strings)


def init_action_list_22( ava_action_list_p ,actionDick):

    labelmap_strings = ''
    with open(ava_action_list_p, 'w',encoding='utf-8') as f:  # 写入action_list文件
        for index, act in tqdm(enumerate(actionDick, start=1), desc='写入action_list文件'):
            labelmap_strings = labelmap_strings + "item {{\n  name: \"{}\"\n  id: {}\n}}\n".format(
                act, index)
        f.write(labelmap_strings)


def rename_rawframes( rawframes_p ):
    dirs = os.listdir(rawframes_p)
    pbar = tqdm(total=len(dirs))
    for dir in dirs:
        pbar.set_description(f'重命名文件夹：{dir}')
        dir_p = os.path.join(rawframes_p, dir)
        imgs = os.listdir(dir_p)

        for img in imgs:

            new_img_name = int(img.split('_')[-1].split('.')[0])
            new_img_name = f'img_{new_img_name:05d}.jpg'
            os.rename(os.path.join(dir_p, img), os.path.join(dir_p, new_img_name))
        pbar.update(1)

def init_frames_list_22(yolo_train_label_p, yolo_val_label_p, frame_list_p):

    if not os.path.exists(frame_list_p):
        os.mkdir(frame_list_p)
    train_list_p = os.path.join(frame_list_p, 'train.csv')
    val_list_p = os.path.join(frame_list_p, 'val.csv')

    video_id = -1
    for label_p in [yolo_train_label_p, yolo_val_label_p]:
        labels = os.listdir(label_p)

        video_id += 1

        if label_p == yolo_train_label_p:
            open_csv = train_list_p
        elif label_p == yolo_val_label_p:
            open_csv = val_list_p
        open_csv_f = open(open_csv, 'w')
        open_csv_f.write('original_vido_id video_id frame_id path labels\n')


        for txt in tqdm(labels[::5], desc=f'加载：{open_csv}'):
            index_index = '_'.join(txt.split('_')[:2])  # 10_1
            original_vido_id = index_index

            for i in range(1, 151):
                frame_id = i
                path = f'{index_index}/img_{frame_id:05d}.jpg'
                open_csv_f.write(f'{original_vido_id} {video_id} {frame_id-1} {path} \"\"\n')
            video_id += 1





def init_ava_21(ann_dir_p, yolo_train_labels_p, yolo_val_labels_p):

    init_ava_files(ann_dir_p, ava21_file_list)

    labelmap_p = os.path.join(ann_dir_p, 'labelmap.txt')
    init_labelmap(labelmap_p)

    init_action_list_21(os.path.join(ann_dir_p, 'ava_action_list_v2.1.pbtxt'), actionDict)

    init_ava_csv(yolo_train_labels_p, os.path.join(ann_dir_p, 'ava_train_v2.1.csv'))
    init_pkl(yolo_train_labels_p, os.path.join(ann_dir_p, 'ava_dense_proposals_train.FAIR.recall_93.9.pkl'))

    init_ava_csv(yolo_val_labels_p, os.path.join(ann_dir_p, 'ava_val_v2.1.csv'))
    init_pkl(yolo_val_labels_p, os.path.join(ann_dir_p, 'ava_dense_proposals_val.FAIR.recall_93.9.pkl'))


def init_ava_22(ann_dir_p, yolo_train_labels_p, yolo_val_labels_p):
    init_ava_files(ann_dir_p, ava22_file_list)

    init_ava_csv(yolo_train_labels_p, os.path.join(ann_dir_p, 'ava_train_v2.2.csv'))
    init_ava_csv(yolo_val_labels_p, os.path.join(ann_dir_p, 'ava_val_v2.2.csv'))

    init_action_list_22(os.path.join(ann_dir_p, 'ava_action_list_v2.2_for_activitynet_2019.pbtxt'), actionDict)
    init_b_t_b_a_l_i_n(yolo_train_labels_p, os.path.join(ann_dir_p, 'ava_detection_train_boxes_and_labels_include_negative_v2.2.csv'))
    init_b_t_b_a_l_i_n(yolo_val_labels_p, os.path.join(ann_dir_p, 'ava_detection_val_boxes_and_labels.csv'), True)

    init_frames_list_22(yolo_train_labels_p, yolo_val_labels_p, os.path.join(ann_dir_p, 'frames_list'))






# ---------------------------初始化ava2.1所需文件---------------------------

annoataion_dir_p = r'H:\AVA\annotations'
train_labels_p = r'H:\AVA\train\labels'
val_labels_p = r'H:\AVA\val\labels'
init_ava_21(annoataion_dir_p, train_labels_p, val_labels_p)
# init_pkl(train_labels_p, os.path.join(annoataion_dir_p, 'ava_dense_proposals_train.FAIR.recall_93.9.pkl'))
# init_pkl(val_labels_p, os.path.join(annoataion_dir_p, 'ava_dense_proposals_val.FAIR.recall_93.9.pkl'))

# ---------------------------初始化ava2.2所需文件---------------------------
# annoataion_dir_p = r'E:\AAA\data\annotation_ava22'
# train_labels_p = r'E:\AAA\data\train\labels'
# val_labels_p = r'E:\AAA\data\val\labels'
# init_ava_22(annoataion_dir_p, train_labels_p, val_labels_p)


# ---------------------------更改所有rawframes文件名img_00001.jpg---------------------------
# rename_rawframes(r'H:\student_action_datat_2\val\rawframes')