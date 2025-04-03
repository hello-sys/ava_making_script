import cv2
import os

from tqdm import tqdm


def extract_frames_from_video(video_path, raw_output_folder, lbimg_output_p, lb_dir_p, dirname, act_id, selected_seconds=5):
    raw_output_folder = os.path.join(raw_output_folder, dirname)
    # 确保输出文件夹存在
    if not os.path.exists(raw_output_folder):
        os.makedirs(raw_output_folder)

    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    # 获取视频的宽度
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 获取视频的高度
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = width // 2
    center_y = height // 2

    # 检查视频是否成功打开
    if not video.isOpened():
        print("无法打开视频文件！")
        return
        # 获取视频的帧率（FPS）
    fps = video.get(cv2.CAP_PROP_FPS)

    # 初始化帧计数器
    frame_count = 1
    old_frame = None
    # 循环读取视频的每一帧
    while frame_count <= fps*selected_seconds:
        ret, frame = video.read()

        # 如果正确读取了帧，ret为True
        if not ret:
            if frame_count<=fps*selected_seconds:
                video.release()
                video = cv2.VideoCapture(video_path)
                continue
            else:
                break

        # 使用帧计数器和五位数字的0填充格式来命名图片
        frame_name = f"img_{frame_count:05d}.jpg"
        frame_path = os.path.join(raw_output_folder, frame_name)
        if frame_count % selected_seconds == 1:
            labe_frame_name = f'{dirname}_{(frame_count//30+1):06d}.jpg'
            lb_frame_p = os.path.join(lbimg_output_p, labe_frame_name)
            cv2.imwrite(lb_frame_p, frame)
            info = [act_id, float(center_x*1.0/width), float(center_y*1.0/height), 1.0, 1.0]
            txt_name = f'{dirname}_{(frame_count//30+1):06d}.txt'
            with open(os.path.join(lb_dir_p, txt_name), 'w') as f:
                f.write(' '.join(map(str, info)))

        # 保存帧为图片
        cv2.imwrite(frame_path, frame)

        # 更新帧计数器
        frame_count += 1

        # 释放视频对象
    video.release()
    # print(f"成功从视频中提取了{frame_count}帧图片！")

def to_ava(info_txt, video_root_p):
    lines = open(info_txt, 'r').readlines()
    act_count_dict = {}
    qbar = tqdm(total=len(lines))
    for line in lines:
        v_p, actid = line.strip().split(' ')
        qbar.set_description(f'{v_p}')
        actid = int(actid)
        act_count_dict[actid] = act_count_dict.get(actid, 0) + 1
        dirname = f'{actid}v_{act_count_dict[actid]}'

        extract_frames_from_video(os.path.join(video_root_p, v_p), raw_p, lbimg_p, lb_dir_p, dirname, actid)
        qbar.update(1)

    for k, v in act_count_dict.items():
        print(f'动作{k}的index: {k}_(1~{v})')



video_path = r"H:\student_action_data_2_ucf_video\videos"  # 替换为你的视频文件路径

save_root = r'H:\student_action_data_2_ucf_video\AVA\val'
raw_p = os.path.join(save_root, 'rawframes')
lbimg_p = os.path.join(save_root, 'labelframes')
lb_dir_p = os.path.join(save_root, 'labels')
if not os.path.exists(raw_p):
    os.makedirs(raw_p)
if not os.path.exists(lbimg_p):
    os.makedirs(lbimg_p)
if not os.path.exists(lb_dir_p):
    os.makedirs(lb_dir_p)

actionDict = {"look down": 0, "look up": 1, "sleep": 2, "phone": 3, "stand": 4, "drink": 5, "raise hand": 6, "chat": 7, "write": 8}

to_ava(r"H:\student_action_data_2_ucf_video\val.txt", video_path)
# extract_frames_from_video(r"H:\student_action_data_2_ucf_video\videos\raisehand\25.mp4", raw_p, lbimg_p, lb_dir_p, '0_1_1', 0)