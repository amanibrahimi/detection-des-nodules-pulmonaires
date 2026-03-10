# 模型预测的相关功能

from Train_Unet import get_unet
import glob
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from skimage import morphology
import os

CHANNEL_COUNT = 1
UNET_WEIGHTS = './model/unet.hd5'
THRESHOLD = 2
BATCH_SIZE = 1
CUBE_SIZE = 32   

def unet_candidate_dicom(unet_result_path):
    centers = []
    image_t = cv2.imread(unet_result_path, cv2.IMREAD_GRAYSCALE)
    image_t[image_t < THRESHOLD] = 0
    image_t[image_t > 0] = 1
    selem = morphology.disk(1)
    image_eroded = morphology.binary_dilation(image_t, selem=selem)
    label_im, nb_labels = ndimage.label(image_eroded)

    for i in range(1, nb_labels + 1):
        blob_i = np.where(label_im == i, 1, 0)
        mass = center_of_mass(blob_i)
        y_px = int(round(mass[0]))
        x_px = int(round(mass[1]))
        centers.append([y_px, x_px])
    return centers


def prepare_image_for_net(img):
    img = img.astype(np.float)
    img /= 255.
    if len(img.shape) == 3:
        img = img.reshape(img.shape[-3], img.shape[-2], img.shape[-1])
    else:
        img = img.reshape(1, img.shape[-2], img.shape[-1], 1)
    return img


def unet_predict(imagepath, maskpath):
    model = get_unet()
    model.load_weights(UNET_WEIGHTS)
    # read png and ready for predict
    images = []
    for files in os.listdir(imagepath):
        tempdir = os.path.join(imagepath, files)
        img = cv2.imread(tempdir, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    for index, img in enumerate(images):
        img = prepare_image_for_net(img)
        images[index] = img
    images3d = np.vstack(images)
    y_pred = model.predict(images3d, batch_size=BATCH_SIZE)
    count = 0
    for y in y_pred:
        y *= 255.
        y = y.reshape((y.shape[0], y.shape[1])).astype(np.uint8)
        cv2.imwrite(os.path.join(maskpath, os.listdir(imagepath)[count]), y)   
        count += 1
    print(count)


def plot_one_box(img, coord, label=None, line_thickness=None):
    """
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.矩形线条粗细
    """
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = [0, 0, 255]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))   
    
    img = cv2.rectangle(img, c1, c2, color, thickness=tl)
    
    return img


if __name__ == "__main__":

    test_img_path = r'G:\DL Project\BiShe\data\Test_images'
    mask_path = r'G:\DL Project\BiShe\data\results'     
    detect_path = r'G:\DL Project\BiShe\data\detect'
    unet_predict(test_img_path, mask_path)
