"""
divide two classes real_frame_files to train, valid and test in same size aligned by first id
"""
import os
import shutil

path_real_files = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Deeperforensics/real_split'
real_files_list = os.listdir(path_real_files)
path_fake_files = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Deeperforensics/fake_split'
fake_files_list = os.listdir(path_fake_files)
path_train_real = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Deeperforensics/train/real/'
path_train_fake = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Deeperforensics/train/fake/'
path_valid_real = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Deeperforensics/valid/real/'
path_valid_fake = '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Deeperforensics/valid/fake/'
path_test_real =  '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Deeperforensics/test/real/'
path_test_fake =  '/home/ncubigdata1/Documents/fanbing_documents_own/DFData/Deeperforensics/test/fake/'

if not os.path.exists(path_train_real):
    os.makedirs(path_train_real)
if not os.path.exists(path_train_fake):
    os.makedirs(path_train_fake)
if not os.path.exists(path_valid_real):
    os.makedirs(path_valid_real)
if not os.path.exists(path_valid_fake):
    os.makedirs(path_valid_fake)
if not os.path.exists(path_test_real):
    os.makedirs(path_test_real)
if not os.path.exists(path_test_fake):
    os.makedirs(path_test_fake)

bound_train = range(0, 599)
bound_valid = range(599, 799)
bound_test = range(799, 999)

for real_file in real_files_list:
    temp = real_file.split('_')[0]
    id_str = temp[:3]
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

