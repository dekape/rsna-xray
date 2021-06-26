
import os
import random
import shutil
import progressbar

EXTENSIONS = ["PNG", "png"]

def is_valid_image(img_path):
    ext = img_path.split(".")[-1]
    if ext in EXTENSIONS:
        return True
    else:
        return False

def get_img_paths(path):
    print("Reading images in %s"%path)

    img_paths = []
    for root, _, fnames in sorted(os.walk(path)):
        for fname in fnames:
            if is_valid_image(fname):
                img_path = os.path.join(root, fname)
                img_paths.append(img_path)
    return img_paths

if __name__ == "__main__":
    covid_root = "/home/dekape/Desktop/COVID-19_Radiography_Dataset/covid"
    covid_paths = get_img_paths((covid_root))
    random.shuffle(covid_paths)

    lo_root = "/home/dekape/Desktop/COVID-19_Radiography_Dataset/lung_opacity"
    lo_paths =  get_img_paths((lo_root))
    random.shuffle(lo_paths)

    pneumonia_root = "/home/dekape/Desktop/COVID-19_Radiography_Dataset/pneumonia"
    pneumonia_paths =  get_img_paths((pneumonia_root)) 
    random.shuffle(pneumonia_paths)
    
    normal_root = "/home/dekape/Desktop/COVID-19_Radiography_Dataset/normal"
    normal_paths =  get_img_paths((normal_root)) 
    random.shuffle(normal_paths)

    ncovid, nlo, npne, nnormal = len(covid_paths), len(lo_paths), len(pneumonia_paths), len(normal_paths)
    train_ratio = 0.95

    train_data_dic = {"covid": covid_paths[:int(ncovid*train_ratio)], 
                     "lung_opacity":lo_paths[:int(nlo*train_ratio)], 
                     "pneumonia":pneumonia_paths[:int(npne*train_ratio)], 
                     "normal":normal_paths[:int(nnormal*train_ratio)] }

    test_data_dic = {"covid": covid_paths[int(ncovid*train_ratio):-1], 
                     "lung_opacity":lo_paths[int(nlo*train_ratio):-1], 
                     "pneumonia":pneumonia_paths[int(npne*train_ratio):-1], 
                     "normal":normal_paths[int(nnormal*train_ratio):-1] }

    test_path = "/home/dekape/Desktop/COVID-19_Radiography_Dataset/test"

    # for path in pneumonia_paths:
    #     path = path.replace(" ", "\ ")
    #     dst_path = path.replace(" ", "_")
    #     os.system("mv %s %s"%(path, dst_path))
    
    
    # reset images
    test_imgs = get_img_paths(test_path)
    for path in test_imgs:
        # path = path.replace(" ", "_")
        img_name = path.split("/")[-1]
        if "COVID" in img_name:
            dst_path = covid_root
        elif "Normal" in img_name:
            dst_path = normal_root
        elif "Viral_Pneumonia" in img_name:
            dst_path = pneumonia_root
        elif "Lung_Opacity" in img_name:
            dst_path = lo_root
        os.system("mv %s %s"%(path, os.path.join(dst_path, img_name)))
        


    for condition in test_data_dic:
        ntrain, ntest = len(train_data_dic[condition]), len(test_data_dic[condition])
        print(condition, "train", ntrain, "test", ntest )
        with progressbar.ProgressBar(max_value=ntest) as bar:
            cnt = 0
            for img_path in test_data_dic[condition]:
                img_name = img_path.split("/")[-1]
                os.system("mv %s %s"%(img_path, os.path.join(test_path,img_name)))

                cnt += 1
                bar.update(cnt)

            print(len(get_img_paths(("/home/dekape/Desktop/COVID-19_Radiography_Dataset/%s"%condition))))
        print("\n")