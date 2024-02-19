import random
import cv2
import tifffile
from PIL import Image
import numpy as np
from keras.models import load_model
from scipy import ndimage
import tensorflow as tf
NUM_TEST_IMAGES = 1024
BS = 32
NUM_EPOCHS = 2
import matplotlib.pyplot as plt

reconstructed_model = load_model("C:\\Users\\DavidS10\\PycharmProjects\\pythonProject\\GPU_run_outputs")
image = tifffile.imread(f'C:\\Users\\DavidS10\\PycharmProjects\\pythonProject\\image_classification\\position3-frame_41.tif')
image_l = tifffile.imread(f'C:\\Users\\DavidS10\\PycharmProjects\\pythonProject\\image_classification\\position3_seg_rotate-frame_41.tif')

rand_int_1 = random.randint(1, 1500)
rand_int_2 = random.randint(1, 1500)
rand_angle = random.choice([0,90,180,270])
# normalization and slicing randomly:
m = np.max(image[:, rand_int_1:rand_int_1 + 256, rand_int_2:rand_int_2 + 256])
print(m)
print(rand_angle)
print(rand_int_1)
print(rand_int_2)
single_image = np.divide(image[:, rand_int_1:rand_int_1 + 256, rand_int_2:rand_int_2 + 256], m)
single_image_label = image_l[1, rand_int_1:rand_int_1 + 256, rand_int_2:rand_int_2 + 256]
single_image_rotate = ndimage.rotate(single_image, rand_angle, axes=(2, 1), reshape=False)
single_image_label_rotate = ndimage.rotate(single_image_label, rand_angle, axes=(1, 0), reshape=False)
tifffile.imsave("image_label.tif",single_image_label_rotate)
tifffile.imsave("image.tif",single_image_rotate)
#single_image_rotate = single_image_rotate.reshape((128, 128, 2))
# single_image = single_image.reshape((128, 128,2))
single_image_label_rotate = single_image_label_rotate.reshape((256, 256))
single_image_label_2channels = np.zeros((2, 256, 256))
single_image_label_2channels[0][single_image_label_rotate == 1] = 1
single_image_label_2channels[1][single_image_label_rotate == 2] = 1
single_image_label_2channels = single_image_label_2channels.transpose((1, 2, 0))
count_of_boundaries = np.count_nonzero(single_image_label_rotate == 0)
norm_count = count_of_boundaries / (256 * 256)
print(norm_count)
tifffile.imsave("one-hot_label_image_0.tif",single_image_label_2channels[:,:,0])
tifffile.imsave("one-hot_label_image_1.tif",single_image_label_2channels[:,:,1])
tifffile.imsave("image_0.tif",single_image_rotate[0,:,:])
tifffile.imsave("image_1.tif",single_image_rotate[1,:,:])
# single_image_label_2channels.reshape((128,128,2))
single_image_rotate= np.transpose(single_image_rotate).reshape((1,256,256,2))
prediction_labels = reconstructed_model.predict(single_image_rotate)
tifffile.imsave("predicted_labels_0.tif",prediction_labels[:,:,:,0])
tifffile.imsave("predicted_labels_1.tif",prediction_labels[:,:,:,1])
single_image_rotate = single_image_rotate[0,:,:,0]
single_image_label_2channels_0 = np.transpose(single_image_label_2channels[:,:,0])
single_image_label_2channels_1 = np.transpose(single_image_label_2channels[:,:,1])
tifffile.imsave("true_labels_0.tif",single_image_label_2channels_0)
tifffile.imsave("true_labels_1.tif",single_image_label_2channels_1)
tifffile.imsave("image_for_labeling.tif",single_image_rotate)

plt.figure()
plt.imshow(prediction_labels[0,:,:,0])
plt.show()
plt.figure()
plt.imshow(prediction_labels[0,:,:,1])
plt.show()
plt.figure()

# make grayscale to black-white
prediction_labels_channel_0 = prediction_labels[:,:,:,0]
prediction_labels_channel_0[prediction_labels_channel_0<0.2] = 0
prediction_labels_channel_0[prediction_labels_channel_0>=0.2] = 1

prediction_labels_channel_1 = prediction_labels[:,:,:,1]
prediction_labels_channel_1[prediction_labels_channel_1<0.8] = 0
prediction_labels_channel_1[prediction_labels_channel_1>=0.8] = 1

tifffile.imsave("black_white_seg_predict_0",prediction_labels_channel_0)
tifffile.imsave("black_white_seg_predict_1",prediction_labels_channel_1)
plt.figure()
plt.imshow(single_image_label_2channels_0)
plt.show()
plt.figure()
plt.imshow(single_image_label_2channels_1)
plt.show()
plt.figure()
plt.imshow(prediction_labels_channel_0[0,:,:])
plt.show()
plt.figure()
plt.imshow(prediction_labels_channel_1[0,:,:])
plt.show()
