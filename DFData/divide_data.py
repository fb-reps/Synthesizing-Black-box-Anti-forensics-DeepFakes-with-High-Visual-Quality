"""
divide two classes real_frame_files to train, valid and test in same size aligned by first id
"""
import os
import shutil

path_real_files = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/align_mp4_real_split/'
real_files_list = os.listdir(path_real_files)
path_fake_files = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/align_mp4_synthesis_split/'
fake_files_list = os.listdir(path_fake_files)
path_train_real = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/train/real/'
path_train_fake = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/train/fake/'
path_valid_real = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/valid/real/'
path_valid_fake = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/valid/fake/'
path_test_real = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/test/real/'
path_test_fake = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Celeb_DF_v2/test/fake/'

if not os.path.exists(path_train_real):
    os.makedirs(path_train_real)
if not os.path.exists(path_train_fake):
    os.makedirs(path_train_fake)

bound_train = range(0, 38)
bound_valid = range(38, 50)
bound_test = range(50, 62)

for real_file in real_files_list:
    temp = real_file.split('_')[0]
    id_str = temp[2:]
    id_int = int(id_str)
    if id_int in bound_train:
        shutil.copyfile(os.path.join(path_real_files, real_file), os.path.join(path_train_real, real_file))
        shutil.copyfile(os.path.join(path_fake_files, real_file), os.path.join(path_train_fake, real_file))
    elif id_int in bound_valid:
        shutil.copyfile(os.path.join(path_real_files, real_file), os.path.join(path_valid_real, real_file))
        shutil.copyfile(os.path.join(path_fake_files, real_file), os.path.join(path_valid_fake, real_file))
    elif id_int in bound_test:
        shutil.copyfile(os.path.join(path_real_files, real_file), os.path.join(path_test_real, real_file))
        shutil.copyfile(os.path.join(path_fake_files, real_file), os.path.join(path_test_fake, real_file))

