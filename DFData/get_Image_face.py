"""
in single thread, extract frames from folder in certain frameFrequency
"""
import dlib  # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2  # 图像处理的库 OpenCv
import os

# 要提取视频的文件名，隐藏后缀
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


path = '/home/ncubigdata1/Documents/fanbing_documents/DFData/Celeb_DF_v2/Celeb_real/'
files = os.listdir(path)
outputDirName = '/home/ncubigdata1/Documents/fanbing_documents/DFData/Celeb_DF_v2/Celeb_real_split/'
if not os.path.exists(outputDirName):
    os.makedirs(outputDirName)

num = 1
for file in files:
    video_path = os.path.join(path, file)
    # print(dir)

    file_name = file.split('.')[0]
    file_name_0 = file_name.split('_')[0]
    file_name_2 = file_name.split('_')[1]

    times = 0
    # 提取视频的频率，每n帧提取一个
    frameFrequency = 60
    # 输出文件到该目录下

    camera = cv2.VideoCapture(video_path)
    # ret, frame = camera.read()

    while True:
        times += 1
        res, img = camera.read()
        if not res:
            print('not res, not image')
            break

        if times % frameFrequency == 0:
            # Dlib 预测器
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
            faces = detector(img, 1)

            for k, face in enumerate(faces):

                # 计算矩形大小
                # (x,y), (宽度width, 高度height)
                # pos_start = tuple([face.left(), face.top()])
                # pos_end = tuple([face.right(), face.bottom()])

                # 计算矩形框大小
                ww = (face.right() - face.left()) // 8
                hh = (face.bottom() - face.top()) // 5
                # img_blank = real_face_image[x1 - ww: x2 + ww, y1 - hh: y2 + hh]
                height = face.bottom() - face.top() + 2 * hh
                width = face.right() - face.left() + 2 * ww

                # 根据人脸大小生成空的图像
                img_blank = np.zeros((height, width, 3), np.uint8)

                for i in range(height):
                    for j in range(width):
                        try:
                            img_blank[i][j] = img[face.top() - hh + i][face.left() - ww + j]

                        except IndexError as e:
                            print('')
                            pass
                # cv2.imshow("face_"+str(num+1), img_blank)
                # img_blank = cv2.resize(img_blank, (256, 256))
                img_blank = cv2.resize(img_blank, (256, 256))

                # 存在本地
                # print("Save to:", outputDirName + file_name + '_{:07d}.png'.format(times))
                cv2.imwrite(outputDirName + file_name + '_{:07d}.png'.format(times), img_blank)

                # cv2.imwrite(outputDirName + file_name + '_{:07d}.png'.format(times), image)
    print('图片提取结束')
    num += 1
    camera.release()

print('ok!!!')
