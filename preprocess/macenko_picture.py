import staintools
import re
import os
import cv2 as cv
import multiprocessing
from pathlib import Path


def image_convert(_img_path, _new_img_path):   
    template_image = Path("n6.png")  
    template_image = str(template_image.resolve())  
    target = staintools.read_image(template_image)  
    normalizer = staintools.StainNormalizer(method='macenko') 
    normalizer.fit(target)  
    image = staintools.read_image(_img_path) 
    image = staintools.LuminosityStandardizer.standardize(image)
    img = normalizer.transform(image)  
    cv.imwrite(_new_img_path, img)  


def create_new_folder(_patient_path, ori_str, replace_str):
    _new_patient_path = re.sub(ori_str, replace_str, _patient_path)
    print(_new_patient_path)
    Path(_new_patient_path).mkdir(parents=True, exist_ok=True)
    return (_new_patient_path)


def color_processing(_root, _img_path, _img):
    _new_img_folder = create_new_folder(_root, "The path of the images to be processed", "The new path of the processed images")   
    _new_img_path = os.path.join(_new_img_folder, _img)  
    image_convert(_img_path, _new_img_path)   
    return 0  


def pre_processing(extra_prefix=""):  
    pool = multiprocessing.Pool(3)    
    _image_path = "The path of the image to be processed" + str(extra_prefix)
    for _root, _dir, _imgs in os.walk(_image_path):
        _imgs = [f for f in _imgs if not f[0] == '.' and f.lower().endswith('.jpg')]   
        _dir[:] = [d for d in _dir if not d[0] == '.']  
        for idx in range(len(_imgs)):   
            _img = _imgs[idx]  
            _img_path = os.path.join(_root, _img)   
            pool.apply_async(color_processing, (_root, _img_path, _img))   #
    pool.close()
    pool.join()


if __name__ == '__main__':
        pre_processing(extra_prefix="/test/0")    #train samples path or test samples path

