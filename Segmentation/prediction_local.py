
import tifffile
import numpy as np
import tensorflow as tf
import os
import skimage as skl



def find_desired_shape(shape_y, shape_x):
    for i in range(shape_y):
        if 2 ** i >= shape_y:
            first_axis_pixels = 2 ** i
            break
    for j in range(shape_x):
        if 2 ** j >= shape_x:
            second_axis_pixels = 2 ** j
            break
    return first_axis_pixels, second_axis_pixels

def normalize_channel(image):
    new_image = np.copy(image)
    per99 = np.percentile(image, 99)
    per1 = np.percentile(image, 1)
    new_image[image > per99] = per99
    new_image[image < per1] = per1
    new_image = new_image - per1
    new_image = new_image/(per99 - per1)
    return new_image

def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPool2D(2)(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)

    # concatenate
    x = tf.keras.layers.concatenate([x, conv_features])
    # dropout
    x = tf.keras.layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


def build_unet_model(InputShape):
    # inputs
    inputs = tf.keras.layers.Input(shape=InputShape)
    f1, p1 = downsample_block(inputs, 128)
    f2, p2 = downsample_block(p1, 256)
    f3, p3 = downsample_block(p2, 512)
    bottleneck = double_conv_block(p3, 1024)
    u1 = upsample_block(bottleneck, f3, 512)
    u2 = upsample_block(u1, f2, 256)
    u3 = upsample_block(u2, f1, 128)
    outputs = tf.keras.layers.Conv2D(2, 1, padding="same", activation="softmax")(u3)
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model

class SegmentationPredictor:

    def __init__(self, model_weights_path, image_shape):
        self.weights_path = model_weights_path
        first_axis_shape, second_axis_shape = find_desired_shape(image_shape[-2], image_shape[
            -1])  # Assuming y and x are the last two axes
        self.model_shape = (first_axis_shape, second_axis_shape, 2)
        self.model = self.initialize_model()
        # print(self.model.summary())  # For debugging


    def initialize_model(self):
        model = build_unet_model(self.model_shape)
        model.load_weights(self.weights_path)
        return model

    def prepare_image(self, image):
        """
        :param image_path: Image should be in axes order (C, Y, X)
        :return:
        """
        # TODO: uncomment after the new training
        tifffile.imwrite("C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\original_img.tif",image)
        normalized_image = np.zeros(image.shape)
        for channel in range(image.shape[0]):
            normalized_image[channel, :, :] = normalize_channel(image[channel, :, :])
        tifffile.imwrite("C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\norm_img.tif",normalized_image)
        # image = image/np.max(image)
        single_image_reshaped = np.transpose(normalized_image).reshape((1, image.shape[2],
                                                                         image.shape[1],
                                                                         image.shape[0]))
        tifffile.imwrite("C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\norm_img_reshapedC0.tif",single_image_reshaped[0,:,:,0])
        tifffile.imwrite(
            "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\norm_img_reshapedC1.tif",
            single_image_reshaped[0, :, :, 1])
        shape1 = single_image_reshaped.shape[1]
        shape2 = single_image_reshaped.shape[2]
        first_axis_pixels, second_axis_pixels = find_desired_shape(shape1, shape2)
        if self.model_shape != (first_axis_pixels, second_axis_pixels, 2):
            self.model_shape = (first_axis_pixels, second_axis_pixels, 2)
            self.initialize_model()
        new_shape = [first_axis_pixels, second_axis_pixels]
        npad = ((0, 0), (new_shape[0] - shape1, 0), (new_shape[1] - shape2, 0), (0, 0))
        padded_image = np.pad(single_image_reshaped, npad)
        tifffile.imwrite("C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\padded_norm_imgC0.tif",padded_image[0,:,:,0])
        tifffile.imwrite(
            "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\padded_norm_imgC1.tif",
            padded_image[0, :, :, 1])
        return padded_image, npad

    def predict(self, image, debug=False):
        padded_image, npad = self.prepare_image(image)
        tifffile.imwrite("C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\ready_to_predictC0.tif",padded_image[0,:,:,0])
        tifffile.imwrite(
            "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\ready_to_predictC1.tif",
            padded_image[0, :, :, 1])
        prediction_labels = self.model.predict(padded_image)
        unpadded_predictions = prediction_labels[:, npad[1][0]:, npad[2][0]:, :]
        tifffile.imwrite("C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\unpadded_predictionsC0.tif",unpadded_predictions[0,:,:,0])
        tifffile.imwrite(
            "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_image_processing\\Segmentation\\unpadded_predictionsC1.tif",
            unpadded_predictions[0, :, :, 1])
        if debug:
            self.model.save(f'outputs/model_image_segmentation_GPU_run')
            print('Saving model history...')
            self.model.save(f'outputs/model.model')
            tifffile.imwrite(f'outputs/predicted_labels_0.tif', prediction_labels[:, :, :, 0])
            tifffile.imwrite(f'outputs/predicted_labels_1.tif', prediction_labels[:, :, :, 1])
            tifffile.imwrite(f'outputs/predicted_labels_0_unpadded.tif',
                unpadded_predictions[:, :, :, 0] * 255)
            tifffile.imwrite(
                f'outputs/predicted_labels_1_unpadded.tif',
                unpadded_predictions[:, :, :, 1] * 255)
            segmentation = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
            SC = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
            SC[unpadded_predictions[0, :, :, 1] > 0.8] = 2
            SC[unpadded_predictions[0, :, :, 0] < 0.1] = 2
            tifffile.imwrite(f'outputs/segmentation_prediction_seg_SC.tif', SC)
            HC = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
            HC[unpadded_predictions[0, :, :, 0] > 0.8] = 1
            HC[unpadded_predictions[0, :, :, 1] < 0.8] = 1
            tifffile.imwrite(f'outputs/segmentation_prediction_seg_HC.tif', HC)
            Boundaries = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
            Boundaries[unpadded_predictions[0, :, :, 0] > 0.2] = 0
            Boundaries[unpadded_predictions[0, :, :, 1] < 0.8] = 0
            tifffile.imwrite(f'outputs/segmentation_prediction_seg_BOUNDARIES.tif', Boundaries)
            seg = HC + SC + Boundaries
            tifffile.imwrite(f'outputs/segmentation_prediction_seg_new.tif', seg)

            segmentation[(unpadded_predictions[0, :, :, 0] > 0.4) & (unpadded_predictions[0, :, :, 1] < 0.8)] = 0
            segmentation[(unpadded_predictions[0, :, :, 0] > 0.8) & (unpadded_predictions[0, :, :, 1] < 0.8)] = 1
            segmentation[(unpadded_predictions[0, :, :, 1] > 0.8) & (unpadded_predictions[0, :, :, 0] < 0.1)] = 2
            tifffile.imwrite(f'outputs/segmentation_prediction_segmentation_new.tif', segmentation)
        HC_B = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
        HC_B[(unpadded_predictions[0, :, :, 0]) > 0.1] = 255
        kernel_dil = np.ones((5, 5), np.uint8)
        img_dilation = skl.morphology.dilation(HC_B, kernel_dil)
        seg_del_eros = skl.morphology.erosion(img_dilation, kernel_dil)
        for i in range(0, 100):
            img_dilation = skl.morphology.dilation(seg_del_eros, kernel_dil)
            seg_del_eros = skl.morphology.erosion(img_dilation, kernel_dil)

        if debug:
            seg_new = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
            seg_new[(seg_del_eros > 0.4) & (unpadded_predictions[0, :, :, 1] < 0.9)] = 0
            seg_new[(seg_del_eros > 0.8) & (unpadded_predictions[0, :, :, 1] < 0.8)] = 1
            seg_new[(unpadded_predictions[0, :, :, 1] > 0.8) & (seg_del_eros < 0.1)] = 2
            tifffile.imwrite(f'outputs/segmentation_prediction_segmentation__dil_ero.tif', seg_new)
            tifffile.imwrite(f'outputs/segmentation_prediction_segmentation_SC.tif', SC)
            seg_new_NEW = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
            # mark boundaries:
            seg_new_NEW[(seg_del_eros > 0.4) & (SC < 0.9)] = 0
            # mark HC:
            seg_new_NEW[(seg_del_eros > 0.8) & (SC < 0.8)] = 1
            # mark SC:
            seg_new_NEW[(SC > 0.8) & (seg_del_eros < 0.1)] = 2
            tifffile.imwrite(f'outputs/segmentation_prediction_segmentation_SEG_NEW.tif', seg_new_NEW)
        HC = skl.morphology.erosion(seg_del_eros, np.ones((7, 7), np.uint8))
        bound = seg_del_eros - HC
        boundary = skl.morphology.dilation(bound, kernel_dil)
        watershed = skl.segmentation.watershed(boundary, watershed_line=True)
        if debug:
            tifffile.imwrite(f'outputs/watershed_img_new.tif', watershed)
            tifffile.imwrite(f'outputs/boundary_img_new.tif', boundary)
            tifffile.imwrite(f'outputs/seg_img_new.tif', segmentation)
        return watershed, HC