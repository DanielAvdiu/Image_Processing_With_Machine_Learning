import os
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip

def load_data(path):
    images = sorted(glob(os.path.join(path, "images", "*.jpg")))
    masks = sorted(glob(os.path.join(path, "masks", "*.jpg")))
    return images, masks
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def augment_data(images, masks, save_path,augment = True):
    H = 256
    W = 256
    for x,y in tqdm(zip(images,masks),total = len(images)):
        name = x.split("\\")[-1].split(".")
        image_name = name[0]
        image_ext = name[1]

        name = y.split("\\")[-1].split(".")
        mask_name = name[0]
        mask_ext = name[1]
        """READING THE IMAGE AND MASK"""
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        """AUGMENTATION"""
        if augment:
            aug = CenterCrop(H, W, p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            save_images = [x, x1, x2, x3, x4, x5]
            save_masks = [y, y1, y2, y3, y4, y5]
        else:
            save_images = [x]
            save_masks = [y]
        """SAVING THE IMAGES AND MASKS"""
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))
            if len(images)==1:
                tmp_image_name = f"{image_name}.{image_ext}"
                tmp_mask_name = f"{mask_name}.{mask_ext}"
            else:
                tmp_img_name = f"{image_name}_{idx}.{image_ext}"
                tmp_mask_name = f"{mask_name}_{idx}.{mask_ext}"
            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            idx += 1

if __name__ == "__main__":
    path = "./DSB"
    images, masks = load_data(path)
    print(f"Original Images: {len(images)} - Original Masks: {len(masks)}")

    """ Creating folders. """
    create_dir("new_data/images")
    create_dir("new_data/masks")

    """Augmenting the data"""
    augment_data(images, masks, "new_data")

    """Loading the augmented data"""
    images, masks = load_data("new_data")
    print(f"Augmented Images: {len(images)} - Augmented Masks: {len(masks)}")