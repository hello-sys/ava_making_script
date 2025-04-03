import os
import re
import shutil
from tqdm import tqdm

dir = 'final_result/yolo/extend/write13'


def rename_folders(directory):
    # 获取目录中的所有文件夹名
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # 对文件夹按字典序排序
    folders.sort()

    # 按顺序重命名文件夹
    for i, folder in enumerate(folders, start=0):
        new_name = f'12_{i}'
        old_path = os.path.join(directory, folder)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed '{folder}' to '{new_name}'")

    # 使用方法：传入要重命名文件夹的目录路径


def rename_images_in_subfolders(root_dir):
    # 正则表达式用于匹配图片文件扩展名（这里假设是.jpg, .png, .jpeg, 你可以根据需要添加更多）
    image_extensions = r'\.(jpg|jpeg|png)$'
    image_pattern = re.compile(image_extensions, re.IGNORECASE)

    # 遍历root_dir下的所有子文件夹
    for subdir, dirs, files in os.walk(root_dir):

        for dir in dirs:
            counter = 1
            # 获取子文件夹名称，并去除路径中的斜杠
            subdir_name = dir
            imgs = [img for img in os.listdir(os.path.join(subdir, dir)) if image_pattern.search(img)]
            subdir_root = os.path.join(subdir, dir)
            for filename in imgs:
                # 检查文件是否是图片
                # 构建新的文件名：子文件夹名_六位序号.扩展名
                new_filename = f"{subdir_name}_{counter:06d}{os.path.splitext(filename)[1]}"
                # 构建旧文件和新文件的完整路径
                old_path = os.path.join(subdir_root, filename)
                new_path = os.path.join(subdir_root, new_filename)
                if not os.path.exists(new_path):
                    # 重命名文件
                    os.rename(old_path, new_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")
                # 更新序号
                counter += 1

                # rename_folders(dir)


def patch_frames(root_dir, seconds, frames_video):
    image_extensions = r'\.(jpg|jpeg|png)$'
    image_pattern = re.compile(image_extensions, re.IGNORECASE)

    total_frames_per_video = seconds * frames_video

    for subdir, dirs, files in os.walk(root_dir):
        img_type = '.jpg'
        for dir in tqdm(dirs):
            imgs = [img for img in os.listdir(os.path.join(subdir, dir)) if image_pattern.search(img)]
            img_count = len(imgs)
            dir_path = os.path.join(subdir, dir)
            if img_count > total_frames_per_video:
                for img in imgs[total_frames_per_video:]:
                    os.remove(os.path.join(dir_path, img))
            elif img_count < total_frames_per_video:
                start_index = img_count + 1
                end_index = total_frames_per_video + 1
                count = 0
                for i in range(start_index, end_index):
                    new_img_name = f"{dir}_{i:06d}{img_type}"
                    new_img = os.path.join(dir_path, new_img_name)
                    if count >= img_count:
                        count = 0
                    old_img = os.path.join(dir_path, imgs[count])
                    count += 1
                    shutil.copy(old_img, new_img)


def select_label_img(source_path, target_path, second, frames_video):
    dirs = os.listdir(source_path)
    for dir in tqdm(dirs):
        dir_path = os.path.join(source_path, dir)
        imgs = os.listdir(dir_path)
        for i in range(0, second):
            img_name = imgs[i * frames_video]
            new_img_name = chang_name(img_name, i+1)
            img_path = os.path.join(dir_path, img_name)
            save_path = os.path.join(target_path, new_img_name)
            shutil.copy(img_path, save_path)
            # print(f"{img_path}---->{save_path}\n")


def chang_name(img_name: str, index: int):
    img_name_wihtou_type = img_name.split('.')[0]
    img_type = img_name.split('.')[1]
    img_new_name = '_'.join(img_name_wihtou_type.split('_')[:-1]) + f"_{index:06d}.{img_type}"
    print(img_new_name)
    return img_new_name

def rename_id_id(old, new_id_id):
    return f"{new_id_id}_{old.split('_')[-1]}"

def rename_img_label_dir(img_p, txt_p, dir_root_p,  chunk_size, start_dir_index):
    imgs = os.listdir(img_p)
    chunk_size = chunk_size # 每次遍历的元素数量
    for i in tqdm(range(0, len(imgs), chunk_size)):
        chunk_imgs = imgs[i:i+chunk_size]
        old_id_id = '_'.join(chunk_imgs[0].split('_')[:2])
        new_id_id = f"71_{start_dir_index}"

        for img in chunk_imgs:

            new_img_name = rename_id_id(img, new_id_id)
            new_txt_name = new_img_name.split('.')[0] + '.txt'

            os.rename(os.path.join(img_p, img), os.path.join(img_p, new_img_name))
            os.rename(os.path.join(txt_p, img.split('.')[0] + '.txt'), os.path.join(txt_p, new_txt_name))
        old_dir_p = os.path.join(dir_root_p, old_id_id)
        old_dir_imgs = os.listdir(old_dir_p)
        for img in old_dir_imgs:
            os.rename(os.path.join(old_dir_p, img), os.path.join(old_dir_p, rename_id_id(img, new_id_id)))
        os.rename(os.path.join(dir_root_p, old_id_id), os.path.join(dir_root_p, new_id_id))

        start_dir_index += 1


def rename_val(val_p, start_first_index):
    old_2_new_name_dict = {}

    labels_dir_p = os.path.join(val_p,'labels')
    labelframes_dir_p = os.path.join(val_p, 'labelframes')
    rawframes_dir_p = os.path.join(val_p, 'rawframes')

    raw_dirs = os.listdir(rawframes_dir_p)

    for dir in tqdm(raw_dirs, desc='建立重命名字典'):
        first_index = dir.split('_')[0]
        if first_index not in old_2_new_name_dict.keys():
            old_2_new_name_dict[first_index] = str(start_first_index)
            start_first_index += 1

    text = []
    for k, v in old_2_new_name_dict.items():
        text.append(f"{k}:{v}")
    text = '\n'.join(text)
    with open(os.path.join(val_p, 'old_2_new.txt'), 'w') as f:
        f.write(text)
    print(text)

    for dir in tqdm(raw_dirs, desc='更改rawframes'):
        first_index = dir.split('_')[0]
        new_dir_name = old_2_new_name_dict[first_index]+'_'+dir.split('_')[-1]
        os.rename(os.path.join(rawframes_dir_p, dir), os.path.join(rawframes_dir_p, new_dir_name))
        old_name_list = [dir + '_000001', dir + '_000002', dir + '_000003', dir + '_000004', dir + '_000005']
        new_name_list = [new_dir_name + '_000001', new_dir_name + '_000002', new_dir_name + '_000003', new_dir_name +
                         '_000004', new_dir_name + '_000005']
        for old_name, new_name in zip(old_name_list, new_name_list):
            os.rename(os.path.join(labelframes_dir_p, old_name+'.jpg'), os.path.join(labelframes_dir_p, new_name+'.jpg'))
            os.rename(os.path.join(labels_dir_p, old_name+'.txt'), os.path.join(labels_dir_p, new_name+'.txt'))


# ==========================批量更改labels和labelframes中文件的前两个index和rawframes中文件夹和文件夹内图片的前两个index==========================
# img_p = r'G:\data\student_action_datat_2\TEMP\drink\labelframes'
# txt_p = r'G:\data\student_action_datat_2\TEMP\drink\labels'
# dir_root_p = r'G:\data\student_action_datat_2\TEMP\drink\rawframes'
# rename_img_label_dir(img_p, txt_p, dir_root_p, 5, 0)    #这里最后一个参数数第二个index，第一个index在代码中更改


# ==========================用于将val整合到train中避免index_index的冲突==========================
rename_val(r'H:\student_action_datat_2\val', 160)

# 文件夹改名
# rename_folders(dir)
# 图片改名
# rename_images_in_subfolders(dir)
# 补齐
# patch_frames(dir, 5, 30)

# select_label_img('final_result/yolo/extend/rawframes', 'final_result/yolo/extend/labelframes', 5, 30)

# rawframes_path = 'final_result/yolo/extend/rawframes'
# labelframes_path = 'final_result/yolo/extend/labelframes'
# select_label_img(rawframes_path, labelframes_path, 5, 30)
# imgs = os.listdir(labelframes_path)
# for img in imgs:
#     # imgs_path = os.path.join(labelframes_path, dir)
#     # imgs = os.listdir(imgs_path)
#     # for img in imgs:
#     img_index = int(img.split('.')[0].split('_')[-1])
#     if img_index > 1:
#         new_index = img_index // 30 + 1
#         new_img_name = chang_name(img, new_index)
#         old_p = os.path.join(labelframes_path, img)
#         new_p = os.path.join(labelframes_path,new_img_name)
#         os.rename(old_p,new_p)

# dirs = os.listdir('final_result/yolo/extend/rawframes')
# for dir in dirs:
#     imgs = os.listdir(os.path.join('final_result/yolo/extend/rawframes', dir))
#     if len(imgs) is not 150:
#         print(f'{dir}-------{len(imgs)}')


