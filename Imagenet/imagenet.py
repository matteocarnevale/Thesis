import os
import requests
import shutil
import tarfile
import json

import os
import requests
import shutil
import tarfile

def download_file(url, target_path, token):
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Download the file
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"Downloaded {target_path}")


def unpack_and_organize(tar_file, target_train_dir, class_json):
    # Load class-to-num mapping from JSON
    num2class = {}
    with open(class_json) as json_file:
        json_data = json.load(json_file)
        for num in json_data:
            num2class[num] = json_data[num][0]  # Map num to class name

    # Unpack the tar.gz file and organize it by class names
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(target_train_dir)
    
    # Organize files into their respective class folders
    for class_zip in sorted(os.listdir(target_train_dir)):
        class_zip_path = os.path.join(target_train_dir, class_zip)
        
        # Check if the file is an archive or image
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
    
    print(f"Organized data in {target_train_dir}")


def download_and_process_dataset(base_url, target_dir, class_json, token, tar_files):
    target_train_dir = os.path.join(target_dir, 'test')

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
    
    # Your Hugging Face token
    token = "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"
    
    # List of tar.gz files to download and process
    tar_files = [
        "test_images.tar.gz"
    ]
    
    # Download, unpack, and organize the dataset
    download_and_process_dataset(base_url, target_dir, class_json, token, tar_files)
