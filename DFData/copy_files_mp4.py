"""
select real_frame_files from second folder which paired with first folder not in place
"""
import os
import shutil

path_pre_1 = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/Celeb_real/'
pre_1_files = os.listdir(path_pre_1)
path_pre_2 = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/Celeb_synthesis/'
pre_2_files = os.listdir(path_pre_2)
path_next_1 = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/align_mp4_Celeb_real/'
path_next_2 = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/align_mp4_Celeb_synthesis/'

if not os.path.exists(path_next_1):
    os.makedirs(path_next_1)
if not os.path.exists(path_next_2):
    os.makedirs(path_next_2)

for pre_1_file in pre_1_files:
    # pre_1_file_path = os.real_face_frame_folder.join(path_A, A_file)
    file_name = pre_1_file.split('.')  # [name , mp4]
    file_name_list = file_name[0].split('_')  # [id_0 , scenes]
    for i in range(0, 62):
        syn_name = "%s_id%d_%s.%s" % (file_name_list[0], i, file_name_list[1], file_name[1])
        if syn_name in pre_2_files:
            shutil.copyfile(os.path.join(path_pre_1, pre_1_file), os.path.join(path_next_1, pre_1_file))
            shutil.copyfile(os.path.join(path_pre_2, syn_name), os.path.join(path_next_2, pre_1_file))
            break
