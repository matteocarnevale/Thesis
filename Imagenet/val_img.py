import os
import requests
import shutil
import tarfile
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

def extract_with_progress(tar_file, target_dir):
    # Extract with progress bar
    with tarfile.open(tar_file, 'r:gz') as tar:
        total_members = len(tar.getmembers())
        with tqdm(total=total_members, desc=f"Extracting {tar_file}", unit="file") as pbar:
            tar.extractall(path=target_dir)
            for member in tar.getmembers():
                pbar.update(1)

def reorganize_val_images(val_dir):
    # The extracted images are typically in the root of `val_dir` or in a subfolder
    val_images_dir = val_dir

    # Iterate over all images in the val folder
    for img_file in sorted(os.listdir(val_images_dir)):
        img_file_path = os.path.join(val_images_dir, img_file)

        # Skip non-JPEG files
        if not img_file.endswith(".JPEG"):
            continue

        # Extract the class name from the filename (e.g., ILSVRC2012_val_00000001_n01751748.JPEG)
        class_name = img_file.split('_')[-1].split('.')[0]

        # Create the class folder inside the val directory
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Move the image into the class folder
        shutil.move(img_file_path, os.path.join(class_dir, img_file))

    print(f"Reorganization complete for {val_dir}")

def download_and_process_val_dataset(base_url, target_dir, token):
    # Create the validation directory
    target_val_dir = os.path.join(target_dir, 'val')

    if not os.path.exists(target_val_dir):
        os.makedirs(target_val_dir)

    # Download and extract val_images.tar.gz
    val_tar_file = os.path.join(target_dir, "val_images.tar.gz")
    val_file_url = f"{base_url}/val_images.tar.gz"

    if not os.path.exists(val_tar_file):
        print(f"{val_tar_file} not found, downloading...")
        download_file(val_file_url, val_tar_file, token)
    else:
        print(f"{val_tar_file} already exists, skipping download.")

    # Extract the validation images
    extract_with_progress(val_tar_file, target_val_dir)

    # Remove the tar.gz file after extraction to save space
    os.remove(val_tar_file)
    print(f"Removed {val_tar_file} to free space")

    # Reorganize validation images into class folders
    reorganize_val_images(target_val_dir)


if __name__ == '__main__':
    # Base URL where your files are hosted (on Hugging Face or another source)
    base_url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data"
    
    # Path to save and organize your dataset
    target_dir = "/home/ubuntu/datasets"
    
    # Ask for the Hugging Face token as input
    token = input("Please enter your Hugging Face token: ")
    
    # Download, unpack, and organize the validation dataset
    download_and_process_val_dataset(base_url, target_dir, token)
