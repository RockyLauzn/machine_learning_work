import os
import requests
import zipfile

# 下载函数
def download_file(url, dest):
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    print(f"Downloaded {dest}")

# 解压函数
def unzip_file(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Unzipped {zip_path} to {dest_dir}")

# 创建目标目录
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"Directory {path} created")

# 数据集下载链接
urls = {
    "annotations": "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "train_images": "https://images.cocodataset.org/zips/train2017.zip",
    "val_images": "https://images.cocodataset.org/zips/val2017.zip"
}

# 目标文件夹路径
base_path = 'C:/Users/rocky/OneDrive/learndata/machine learning/bigwork/CNNH/CNNH/data/coco'
annotations_path = os.path.join(base_path, 'annotations')
train_images_path = os.path.join(base_path, 'train2017')
val_images_path = os.path.join(base_path, 'val2017')

# 创建目标文件夹
create_directory(annotations_path)
create_directory(train_images_path)
create_directory(val_images_path)

# 下载文件并解压
download_file(urls["annotations"], 'annotations_trainval2017.zip')
unzip_file('annotations_trainval2017.zip', annotations_path)

download_file(urls["train_images"], 'train2017.zip')
unzip_file('train2017.zip', train_images_path)

download_file(urls["val_images"], 'val2017.zip')
unzip_file('val2017.zip', val_images_path)

# 删除下载的压缩文件
os.remove('annotations_trainval2017.zip')
os.remove('train2017.zip')
os.remove('val2017.zip')

print("All files downloaded and unzipped successfully.")
