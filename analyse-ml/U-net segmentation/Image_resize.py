import cv2   
import os   

size_ = (320, 320)
 

directory = r'D:/Image/'

image_to_save_directory = os.path.join(directory, 'resized_image/')
if not os.path.isdir(image_to_save_directory):
    os.mkdir(image_to_save_directory)
file_names = os.listdir(directory)
# print(file_names)


for filename in file_names:
    file_path = os.path.join(directory, filename)
    # print(file_path)
    img = cv2.imread(file_path)
    im2 = cv2.resize(img, size_, interpolation=cv2.INTER_CUBIC)

    saved_path = os.path.join(image_to_save_directory, filename)
    cv2.imwrite(saved_path, im2)

print(' image  resized successfully')
