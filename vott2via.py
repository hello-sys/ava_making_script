# import pandas as pd
# data = pd.read_csv("vott_student_action-export.csv")

import csv
from tqdm import tqdm

via_header = ['metadata_id', 'file_list', 'flags', 'temporal_coordinates', 'spatial_coordinates', 'metadata']
# "1_xUaXczHh","[""1_0_00001.jpg""]",0,"[]","[2,915,227,159,194]","{""1"":""5""}"

metadata_id = '""'
file_list = '"[""{}""]"'
flags = '0'
temporal_coordinates = '"[]"'
spatial_coordinates = '"{}"'
metadata = '"{""1"":""""}"'
# print(metadata.)
# {"0":"低头","1":"抬头","2":"睡觉","3":"玩手机","4":"站立","5":"喝水","6":"举手","7":"交谈"}
action_dict = {'look down': 0, 'look up': 1, 'sleep': 2, 'phone': 3, 'stand': 4, 'drink': 5, 'raise hand': 6, 'chat': 7, 'write': 8 }
with open('final_result/yolo/extend/extend_drink_phone_write-export.csv', 'r', encoding='utf-8', newline='') as vott:
    csv_vott = csv.DictReader(vott)
    header = csv_vott.fieldnames    #"image","xmin","ymin","xmax","ymax","label"
    print(header)
    for vott_row in tqdm(list(csv_vott)):
        file_list.format(vott_row['image'])
        sc = [2]
        x = round(float(vott_row["xmin"]))
        y = round(float(vott_row["ymin"]))
        w = round( float(vott_row["xmax"]) - float(vott_row["xmin"]) )
        h = round( float(vott_row["ymax"]) - float(vott_row["ymin"]) )
        sc.append(x)
        sc.append(y)
        sc.append(w)
        sc.append(h)
        file_list = '"[""{}""]"'.format(vott_row['image'])
        spatial_coordinates = '"{}"'.format(sc).replace(' ', '')
        action_id = action_dict[vott_row['label']]
        metadata = '"{""1"":""' + str(action_id) + '""}"'
        via_row = metadata_id+','+file_list+','+flags+','+temporal_coordinates+','+spatial_coordinates+','+metadata
        # print(via_row)
        with open('final_result/yolo/extend/via_drink_phone_write.csv', 'a+', encoding='utf-8', newline='') as via:
            via.write(via_row+'\n')
