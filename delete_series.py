import shutil
import os

from tqdm import tqdm


def delet_series(delete_index_index_list, labelframe_p, label_p, rawframes_root_p):
    pbar = tqdm(total=len(delete_index_index_list))
    for delete_index_index in delete_index_index_list:
        delete_list = [delete_index_index + '_000001', delete_index_index + '_000002', delete_index_index + '_000003',
                       delete_index_index + '_000004', delete_index_index + '_000005']  # 118_5_000002
        for d_name in delete_list:
            if not os.path.exists(os.path.join(labelframe_p, d_name + '.jpg')):
                print('文件不存在：' + os.path.join(labelframe_p, d_name + '.jpg'))
                continue
            os.remove(os.path.join(labelframe_p, d_name + '.jpg'))
            os.remove(os.path.join(label_p, d_name + '.txt'))
            pbar.set_description(f"删除 {d_name}.jpg,{d_name}.txt")
            # print('已删除：' + os.path.join(labelframe_p, d_name + '.jpg') + '\t' + os.path.join(labelframe_p, d_name + '.jpg'))
            # print('已删除：' + os.path.join(label_p, d_name + '.txt') + '\t' + os.path.join(label_p, d_name + '.txt'))
        if not os.path.exists(os.path.join(rawframes_root_p, delete_index_index)):
            pbar.update(1)
            print('文件不存在：' + os.path.join(rawframes_root_p, delete_index_index))
            continue
        shutil.rmtree(os.path.join(rawframes_root_p, delete_index_index))
        pbar.set_description(f"删除 {delete_index_index}文件夹")
        pbar.update(1)

def cut_series(cut_index_index_list, labelframe_p, label_p, rawframes_root_p, target_save_p):

    label_img_save_p = os.path.join(target_save_p, 'labelframes')
    label_save_p = os.path.join(target_save_p, 'labels')
    rawframes_save_p = os.path.join(target_save_p, 'rawframes')
    if not os.path.exists(label_img_save_p):
        os.makedirs(label_img_save_p)
    if not os.path.exists(label_save_p):
        os.makedirs(label_save_p)
    if not os.path.exists(rawframes_save_p):
        os.makedirs(rawframes_save_p)
    pbar = tqdm(total=len(cut_index_index_list))
    for cut_index_index in cut_index_index_list:
        cut_list = [cut_index_index + '_000001', cut_index_index + '_000002', cut_index_index + '_000003',
                       cut_index_index + '_000004', cut_index_index + '_000005']  # 118_5_000002
        for d_name in cut_list:
            pbar.set_description(f"移动 {d_name}.jpg,{d_name}.txt")
            if not os.path.exists(os.path.join(labelframe_p, d_name + '.jpg')):
                print("不存在："+os.path.join(labelframe_p, d_name + '.jpg'))
                break
            try:
                if not os.path.exists(os.path.join(label_img_save_p, d_name + '.jpg')):
                    shutil.move(os.path.join(labelframe_p, d_name + '.jpg'), os.path.join(label_img_save_p, d_name + '.jpg'))
                if not os.path.exists(os.path.join(label_save_p, d_name + '.txt')):
                    shutil.move(os.path.join(label_p, d_name + '.txt'), os.path.join(label_save_p, d_name + '.txt'))
            except OSError as e:
                print(f"在移动文件时发生错误: {e.strerror}")
        try:
            if not os.path.exists(os.path.join(rawframes_root_p, cut_index_index)):
                print("不存在文件夹："+os.path.join(rawframes_root_p, cut_index_index))
            else:
                if not os.path.exists(os.path.join(rawframes_save_p, cut_index_index)):
                    shutil.move(os.path.join(rawframes_root_p, cut_index_index), os.path.join(rawframes_save_p, cut_index_index))
                    pbar.set_description(f"移动 {cut_index_index}文件夹")
        except OSError as e:
            print(f"在移动文件夹时发生错误: {e.strerror}{os.path.join(rawframes_save_p, cut_index_index)}")
        pbar.update(1)

root = r'H:\student_action_datat_2\train'
labelframe_p = f"{root}/labelframes"
label_p = f"{root}/labels"
rawframes_root_p = f"{root}/rawframes"
#
# save_p = r"G:\data\student_action_datat_2\val"
#
# delete_index_index_list = {126:128}
# new_index_index = []
# for k, v in delete_index_index_list.items():
#     for i in range(v+1):
#         new_index_index.append(f'{k}_{i}')
#
# # index = [71,84]
# # new_index_index = ['70_'+str(i) for i in index ]
# # cut_index_index_list, labelframe_p, label_p, rawframes_root_p, target_save_p
# cut_series(new_index_index, labelframe_p, label_p, rawframes_root_p, save_p)

# lookup = [4,6,10,11,12,13,14,16,18,21,23,24,25,28,29,34,37,38,39,40,48,52,53,54,57,61,62,64,65,68,69,72,70,73,74,78,89,93,94,96,99,
#          101,110,113,114,116,120,124,126,129,130,149,161,164,166,169,171,173,174,181,184,186,188,191,193,194,196,198,209,214,217,219,220,
#          221,226,236,237,239,248,251,253,254,256,259,262,263,265,272,276,277,287,291,292,297,303,339,344,347,366,376,381,398,404,420,421,427,
#          428,436,440,450,459,461,463,469,470,477,485,435,547,570,584,590,602,614,617,619,624,641,643,645,648,650,662,670,676,686,720,829,981,
#          995,1021,1022,1026,1027,1031,1032,1042,1050,1053,1072,1126,1217,1221,1233,1313,1358,1365,1377,1408,1432,1470,1510,1525,1530,1533,1556,
#          1564,1596,1631,1637,1658,1657,1663,1664,1669,1674,1726,1734,1751,1757,1811,1821,1828,1836,1904,1924,1934,1938,1944,2068,2148,2164,
#          2474,2491,2539,2574,2626,2726,2748,2821,2932,3085,3456,3493,3499,3517,3531,3532,3534,3544,3554,3559,3568,3570,3581,3534,3697,3702,3720,3721,
#          3724,3726,3730,3733,3744,3747,3748,3749,3753,3758,3763,3765,3766,3797,3810,3811,3812,3821,3823,3824,3830,3837,3853,3930,3958,3969,4010,4042,4135,
#          4139]
# chat = [12,365,368,372,375,376,398,407,413,418,420,421,422,423,424,471,472,474,476,485,489,490,493,576,579,582,585,591,]
index = list(range(0,661))
new_index_index = ['113_'+str(i) for i in index ]
delet_series(new_index_index, labelframe_p, label_p, rawframes_root_p)
