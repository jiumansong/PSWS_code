
import os
import tqdm
import pandas as pd
from sarcoma import open_slide


def tiling(_label_file, _test_file, _tile_size=2016, _tile_level=5):

    _train_tile_base = "train_samples_patch_out_path"
    _test_tile_base = "test_samples_patch_out_path"

    if not os.path.exists(_train_tile_base):
        os.mkdir(_train_tile_base)
    if not os.path.exists(_test_tile_base):
        os.mkdir(_test_tile_base)

    _format = "jpg"
    _tile_size = _tile_size-2 
    _overlap = 1  
    _limit_bounds = True
    _quality = 90  #The quality of the patch images
    _workers = 12  

    if _label_file is not None:
        print("Training WSIs: start tiling ...")
        _label_df = pd.read_csv(_label_file)
        _svs_path = _label_df.iloc[:, 0]
        _svs_label = _label_df.iloc[:, 1]  
        for i in tqdm.tqdm(range(len(_svs_path))):
            _curr_svs = _svs_path.iloc[i]  
            _folder_name = os.path.join(_train_tile_base, '\\'.join(_curr_svs.split("\\")[-2:]).split(".")[0]) 
            print(_folder_name)
            open_slide.DeepZoomStaticTiler(_curr_svs, _folder_name, _format,
                                           _tile_size, _overlap, _limit_bounds, _quality,
                                           _workers, _tile_level).run()
            print("one slide file is done")
        print("The number of train slide is:", len(_svs_path))
        print("All train slide are done")
    else:
        print("No training WSI is provided.")

    if _test_file is not None:
        print("Testing WSIs: start tiling ...")
        _label_df = pd.read_csv(_test_file)
        _svs_path = _label_df.iloc[:, 0]
        _svs_label = _label_df.iloc[:, 1]

        for i in tqdm.tqdm(range(len(_svs_path))):
            _curr_svs = _svs_path.iloc[i]
            _folder_name = os.path.join(_test_tile_base, '\\'.join(_curr_svs.split("\\")[-2:]).split(".")[0])
            open_slide.DeepZoomStaticTiler(_curr_svs, _folder_name, _format,
                                           _tile_size, _overlap, _limit_bounds, _quality,
                                           _workers, _tile_level).run()
        tmp_dict = {"Image_path": new_path, "Label": new_label}
        print("The number of test slide is:", len(_svs_path))
        print("All test slide are done")

    else:
        print("No testing WSI is provided.")

if __name__ == '__main__':
        tiling('train_slide_csv_path.csv',      #the pait of slide_path and label in csv
               'test_slide_csv_path.csv', _tile_size=2016, _tile_level=5)
        #_tile_level is objective magnification