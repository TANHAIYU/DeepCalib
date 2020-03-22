import numpy as np
import cv2
import time, glob
import random
import skimage
import distortion_model

random.seed(9002)   # 9001
np.random.seed(2)   #1

# path_to_images = '/home/haiyutan/master-thesis/images/Dataset/Dataset_512_ori/*.png' #seq 00
# path_to_images = '/mnt/usb-Seagate_Expansion_NAA9EWS9-0:0-part1/Dataset/dataset/sequences/21/image_2/*.png' #seq21
path_to_images = '/mnt/usb-Seagate_Expansion_NAA9EWS9-0:0-part1/Dataset/dataset/sequences/20/image_2/*.png' #seq
# path_to_images = '/mnt/usb-Seagate_Expansion_NAA9EWS9-0:0-part1/Dataset/dataset/sequences/19/image_2/*.png' #seq
# path_to_images = '/mnt/usb-Seagate_Expansion_NAA9EWS9-0:0-part1/Dataset/dataset/sequences/18/image_2/*.png' #seq
# path_to_images = '/mnt/usb-Seagate_Expansion_NAA9EWS9-0:0-part1/Dataset/dataset/sequences/17/image_2/*.png' #seq
# path_to_images = '/mnt/usb-Seagate_Expansion_NAA9EWS9-0:0-part1/Dataset/dataset/sequences/16/image_2/*.png' #seq
# path_to_images = '/mnt/usb-Seagate_Expansion_NAA9EWS9-0:0-part1/Dataset/dataset/sequences/15/image_2/*.png' #seq
# path_to_images = '/mnt/usb-Seagate_Expansion_NAA9EWS9-0:0-part1/Dataset/dataset/sequences/14/image_2/*.png' #seq

list_image_paths = glob.glob(path_to_images)

starttime = time.clock()
sum = 0
for image_path in list_image_paths:  # length of your filename list
	print("Processing",image_path)

	width = 1241
	height = 376

	parameters = distortion_model.distortionParameter()
	f = parameters[3]
	Lambda = parameters[0]
	x0    = parameters[1]  # princip point - Cx
	y0    = parameters[2]  # princip point - Cy

	OriImg = cv2.imread(image_path)
	# temImg = rescale(OriImg, 0.5, mode='reflect',anti_aliasing= True) #0.5
	temImg = cv2.resize(OriImg, dsize=(1241, 376), interpolation=cv2.INTER_CUBIC)
	ScaImg = skimage.img_as_ubyte(temImg)

	padImg = np.array(np.zeros((ScaImg.shape[0] + 1, ScaImg.shape[1] + 1, 3)), dtype=np.uint8)
	padImg[0:height, 0:width, :] = ScaImg[0:height, 0:width, :]
	padImg[height, 0:width, :] = ScaImg[height - 1, 0:width, :]
	padImg[0:height, width, :] = ScaImg[0:height, width - 1, :]
	padImg[height, width, :] = ScaImg[height - 1, width - 1, :]

	disImg = np.array(np.zeros(ScaImg.shape), dtype=np.uint8)
	u = np.array(np.zeros((ScaImg.shape[0], ScaImg.shape[1])), dtype=np.float32)
	v = np.array(np.zeros((ScaImg.shape[0], ScaImg.shape[1])), dtype=np.float32)

	for i in range(width):
		for j in range(height):

			xu, yu = distortion_model.distortionModel(i, j, width, height, parameters)

			if (0 <= xu <= width - 1) and (0 <= yu <= height - 1):
				u[j][i] = xu - i
				v[j][i] = yu - j

				# Bilinear interpolation
				Q11 = padImg[int(yu), int(xu), :]
				Q12 = padImg[int(yu), int(xu) + 1, :]
				Q21 = padImg[int(yu) + 1, int(xu), :]
				Q22 = padImg[int(yu) + 1, int(xu) + 1, :]

				disImg[j, i, :] = Q11 * (int(xu) + 1 - xu) * (int(yu) + 1 - yu) + Q12 * (xu - int(xu)) * (int(yu) + 1 - yu) + \
				                  Q21 * (int(xu) + 1 - xu) * (yu - int(yu)) + Q22 * (xu - int(xu)) * (yu - int(yu))

	name = image_path.split('/')[-1]
	name_list = name.split('.')
	sum = sum + 1

	EndImg = cv2.resize(disImg, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
	print(str(sum) + "/" +str(len(list_image_paths)) + "th image finished")
	cv2.imwrite('/home/haiyutan/master-thesis/images/dataset/inceptionv3_test_discrete_add_dc_distortionCenter/'+ name_list[0] + '_Lam_' + str(Lambda) + '_f_' + str(f) +'_dc_'+ str(x0)+ '_'+str(y0)+'.' + name_list[ -1], EndImg)
print "elapsed time ", time.clock() - starttime