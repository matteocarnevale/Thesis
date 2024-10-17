import os
import requests
import shutil
import tarfile
import json
from tqdm import tqdm  # For progress bars
import concurrent.futures  # For parallelism

def download_file(url, target_path, token):
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Stream the download with progress bar
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

def extract_with_progress(tar_file, target_train_dir):
    # Extract with progress bar
    with tarfile.open(tar_file, 'r:gz') as tar:
        total_members = len(tar.getmembers())
        with tqdm(total=total_members, desc=f"Extracting {tar_file}", unit="file") as pbar:
            tar.extractall(path=target_train_dir)
            for member in tar.getmembers():
                pbar.update(1)

def organize_files(class_zip_path, target_train_dir, num2class):
    # Organize files into their respective class folders
    class_zip = os.path.basename(class_zip_path)
    if class_zip.endswith(".tar") or class_zip.endswith(".tar.gz"):
        class_, _ = class_zip.split('.')
        class_dir = os.path.join(target_train_dir, num2class.get(class_, class_))  # Create class directory
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        shutil.unpack_archive(class_zip_path, class_dir)  # Extract the archive to the class directory
        os.remove(class_zip_path)  # Remove the archive after extraction
    elif class_zip.endswith(".JPEG"):
        class_, _ = class_zip.split('_', 1)  # Split on first underscore to get class name
        class_dir = os.path.join(target_train_dir, class_)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        shutil.move(class_zip_path, class_dir)  # Move the image to the class directory

def unpack_and_organize(tar_file, target_train_dir, class_json):
    # Load class-to-num mapping from JSON
    num2class = {}
    with open(class_json) as json_file:
        json_data = json.load(json_file)
        for num in json_data:
            num2class[num] = json_data[num][0]  # Map num to class name

    # Step 1: Extract tar.gz file
    extract_with_progress(tar_file, target_train_dir)

    # Step 2: Organize files in parallel using multi-threading
    class_zip_paths = [os.path.join(target_train_dir, file) for file in os.listdir(target_train_dir)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(organize_files, class_zip_path, target_train_dir, num2class) for class_zip_path in class_zip_paths]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure all tasks are completed

    print(f"Organized data in {target_train_dir}")

def download_and_process_dataset(base_url, target_dir, class_json, token, tar_files):
    target_train_dir = os.path.join(target_dir, 'train')

    # Process each tar.gz file one by one
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
    # Base URL where your files are hosted (on Hugging Face or another source)
    base_url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data"
    
    # Path to save and organize your dataset
    target_dir = "/media/HDD/carnevale/datasets/imaget1k"
    
    # Path to the JSON file mapping class numbers to class names
    class_json = "/media/HDD/carnevale/datasets/imaget1k/ImageNet_class_index.json"
    
    # Ask for the Hugging Face token as input
    token = input("Please enter your Hugging Face token: ")
    
    # List of tar.gz files to download and process
    tar_files = [
        "train_images_0.tar.gz",
        "train_images_1.tar.gz",
        "train_images_2.tar.gz",
        "train_images_3.tar.gz",
        "train_images_4.tar.gz",
    ]
    
    # Download, unpack, and organize the dataset
    download_and_process_dataset(base_url, target_dir, class_json, token, tar_files)
