"""
pick first id in [0,38) and second id no limit fake videos to a new folder
"""
import os
import shutil

path_fake_files = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/Celeb_synthesis/'
fake_files_list = os.listdir(path_fake_files)
path_synthesis_videos = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/synthesis_38_49_videos/'

if not os.path.exists(path_synthesis_videos):
    os.makedirs(path_synthesis_videos)

bound = range(38, 50)

for fake_file in fake_files_list:
    temp = fake_file.split('_')[0]
    id_str = temp[2:]
    id_int = int(id_str)
    if id_int in bound:
        shutil.copyfile(os.path.join(path_fake_files, fake_file), os.path.join(path_synthesis_videos, fake_file))
