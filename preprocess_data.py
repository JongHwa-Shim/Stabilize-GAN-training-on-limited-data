import os
from PIL import Image
import random
import shutil

import torchvision.transforms as transforms

from setting import config

def preprocessing(file_path, save_root, data_shape, option=None): # image load , crop, resize and save 
    img = Image.open(file_path)
    width, height = img.size
    
    if option == 'center_crop':
        if width > height:
            crop = transforms.CenterCrop(height, height)
        else:
            crop = transforms.CenterCrop(width, width)
        img = crop(img)
    elif option == 'default':
        pass
    else:
        print('please select option')

    img_resize = img.resize(data_shape)
    img_resize.save(os.path.join(save_root, os.path.basename(file_path)))

if __name__ == '__main__':
    cfg = config()

    data_dir = 'D:/DATA_ARCHIVE/DATASET/lsun/dog'
    data_shape = (256,256)
    new_data_root = 'C:/Code/Stabilize-GAN-training-on-limited-data/dataset'
    data_name = 'LSUN_dog'

    data_num_list = [100, 1000, 10000, 50000]

    data_list = []
    for (root, dirs, files) in os.walk(data_dir):
        if len(files) is not 0:
            for file in files:
                data_list.append(os.path.join(root, file))
    
    for data_num in data_num_list:
        save_root = os.path.join(new_data_root, data_name + '_' + str(data_num))

        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.makedirs(save_root)

        sampled_files = random.sample(data_list, data_num)
        for sampled_file in sampled_files:
            preprocessing(sampled_file, save_root, data_shape, option='default')
    




    

    



    
