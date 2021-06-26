
import os
import random
import pandas as pd

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
    random.seed(33)
    
    test_path = "C:\\Users\\debor\\Google Drive\\IC_PHD\\GTA\ACSE20-21\\ACSE4\\xray-data\\test"
    
    key_file = os.path.join("./", "test_key.csv")
    key_annotated_file = os.path.join("./", "test_key_annotated.csv")
    
    test_imgs = get_img_paths(test_path)
    random.shuffle(test_imgs)
    
    df = pd.DataFrame(columns=["name", "target", "original_name"])
    for i, path in enumerate(test_imgs):
        root_path = "\\".join(path.split("\\")[:-1])
        original_name = path.split("\\")[-1].split(".")[0]
        new_name = "test_"+str(i)
        dst_path = os.path.join(r"{}".format(root_path), new_name+".png")
        
        if "COVID" in original_name: target = 0
        elif "Lung_Opacity" in original_name: target = 1
        elif "Viral_Pneumonia" in original_name: target = 2
        elif "Normal" in original_name: target = 3
            
        df.loc[i] = [new_name, target, original_name]

        # win32 bullshit
        
        dst_path = dst_path.replace("\\", "\\\\")
        path = path.replace("\\", "\\\\")
        os.system("mv \'{}\' \'{}\'".format(path, dst_path))
        
        print(original_name, "------------>", new_name)
        
    df.to_csv(key_annotated_file, index=False)
    df = df.drop(columns=["original_name"])
    df.to_csv(key_file, index=False)
        
        