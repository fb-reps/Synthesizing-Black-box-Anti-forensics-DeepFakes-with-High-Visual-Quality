"""
in two folders, delete unaligned frame which not have the same frame name in another folder in place
"""
import os

path_A = '/run/user/1000/gvfs/smb-share:server=user-z490-taichi.local,share=lsq_workspace_sharpen/SSIM_PSNR_project/data/Deeperforensics/OursClip'
A_files = os.listdir(path_A)
path_B = '/run/user/1000/gvfs/smb-share:server=user-z490-taichi.local,share=lsq_workspace_sharpen/SSIM_PSNR_project/data/Deeperforensics/DingClip'
B_files = os.listdir(path_B)

for A_file in A_files:
    if A_file not in B_files:
        delete_file = os.path.join(path_A, A_file)
        os.remove(delete_file)

for B_file in B_files:
    if B_file not in A_files:
        delete_file = os.path.join(path_B, B_file)
        os.remove(delete_file)

A_files_num = os.path.getsize(path_A)
B_files_num = os.path.getsize(path_B)
print("path_A: %s\n num: %d\n path_B: %s\n num: %d\n" % (path_A, A_files_num, path_B, B_files_num))
