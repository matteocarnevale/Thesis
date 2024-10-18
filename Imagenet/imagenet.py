import os
import requests
import shutil
import tarfile
import json
import subprocess
from tqdm import tqdm  # For progress bars

def download_file(url, target_path, token):
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Stream the download with a progress bar
    with requests.get(url, headers=headers, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(target_path, 'wb') as f, tqdm(
            desc=target_path, total=total_size, unit='B', unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print(f"Downloaded {target_path}")

def extract_tar_file(tar_file, target_dir, num_threads=30):
    """
    Extracts a .tar.gz file using pigz and tar for multi-threaded decompression, with a progress bar using pv.
    The block sizes for pv and pigz are set to 4 MB.

    Args:
        tar_file (str): The path to the .tar.gz file.
        target_dir (str): The directory to extract the contents to.
        num_threads (int): The number of threads to use for decompression.
    """
    print(f"Extracting {tar_file} to {target_dir} with pigz using {num_threads} threads, 4MB block size, and a progress bar...")

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Set block size to 4MB (4M) for both pv and pigz
    try:
        command = f"pv -B 16M {tar_file} | pigz -b 16384 -p {num_threads} -dc | tar xf - -C {target_dir}"
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during extraction: {e}")
    else:
        print(f"Successfully extracted {tar_file} to {target_dir}")



def organize_files(class_zip_path, target_dir, num2class):
    # Organize files into their respective class folders
    class_zip = os.path.basename(class_zip_path)
    if class_zip.endswith(".tar") or class_zip.endswith(".tar.gz"):
        class_name = class_zip.split('.')[0]
        class_dir = os.path.join(target_dir, num2class.get(class_name, class_name))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        shutil.unpack_archive(class_zip_path, class_dir)
        os.remove(class_zip_path)  # Remove the archive after extraction

    elif class_zip.endswith(".JPEG"):
        class_name = class_zip.split('_', 1)[0]
        class_dir = os.path.join(target_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        shutil.move(class_zip_path, class_dir)


def unpack_and_organize(tar_file, target_dir, class_json):
    # Load class-to-num mapping from JSON
    with open(class_json) as json_file:
        num2class = {num: data[0] for num, data in json.load(json_file).items()}

    # Step 1: Extract tar.gz file
    extract_tar_file(tar_file, target_dir, 24)

    # Step 2: Organize files in the extracted folder
    class_zip_paths = [os.path.join(target_dir, f) for f in os.listdir(target_dir)]
    for class_zip_path in class_zip_paths:
        organize_files(class_zip_path, target_dir, num2class)

def download_and_process_dataset(base_url, target_dir, class_json, token, tar_files):
    target_train_dir = os.path.join(target_dir, 'train')

    for tar_file in tar_files:
        tar_file_url = f"{base_url}/{tar_file}"
        local_tar_file = os.path.join(target_dir, tar_file)

        # Step 1: Check if the tar.gz file is already downloaded
        if not os.path.exists(local_tar_file):
            print(f"{local_tar_file} not found, downloading...")
            download_file(tar_file_url, local_tar_file, token)
        else:
            print(f"{local_tar_file} already exists, skipping download.")

        # Step 2: Unpack and organize it
        unpack_and_organize(local_tar_file, target_train_dir, class_json)
        
        # Step 3: Remove the tar.gz file to save space
        os.remove(local_tar_file)
        print(f"Removed {local_tar_file} to free space")

if __name__ == '__main__':
    base_url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data"
    target_dir = "/home/ubuntu/datasets/imagenet"
    class_json = "/home/ubuntu/Thesis/Imagenet/ImageNet_class_index.json"
    token = input("Please enter your Hugging Face token: ")

    tar_files = [
        "train_images_0.tar.gz",
        "train_images_1.tar.gz",
        "train_images_2.tar.gz",
        "train_images_3.tar.gz",
        "train_images_4.tar.gz",
    ]
    
    download_and_process_dataset(base_url, target_dir, class_json, token, tar_files)
