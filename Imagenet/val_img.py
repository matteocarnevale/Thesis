import os
import shutil

def reorganize_val_images(val_dir):
    ilsvrc_dir = os.path.join(val_dir, 'ILSVRC2012')

    # Check if the ILSVRC2012 folder exists
    if not os.path.exists(ilsvrc_dir):
        print(f"{ilsvrc_dir} does not exist. Exiting.")
        return

    # Iterate over all images in the ILVSRC2012 folder
    for img_file in sorted(os.listdir(ilsvrc_dir)):
        img_file_path = os.path.join(ilsvrc_dir, img_file)

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

    # Remove the empty ILVSRC2012 folder
    shutil.rmtree(ilsvrc_dir)
    print(f"Removed the folder {ilsvrc_dir}. Reorganization complete.")

if __name__ == '__main__':
    # Path to the val directory
    val_dir = "/media/HDD/carnevale/datasets/imaget1k/test"

    # Reorganize validation images
    reorganize_val_images(val_dir)
