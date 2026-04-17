import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class CarlaDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        self.transform = transform
        self.samples = []

        for d in img_dirs:
            rgb_dir = os.path.join(d, "CameraRGB")
            seg_dir = os.path.join(d, "CameraSeg")

            images = sorted(os.listdir(rgb_dir))
            masks = sorted(os.listdir(seg_dir))

            for img_name, mask_name in zip(images, masks):
                self.samples.append((
                    os.path.join(rgb_dir, img_name),
                    os.path.join(seg_dir, mask_name)
                ))

    def __len__(self):
        return len(self.samples)

    def process_mask(self, mask):
        return mask[:, :, 2]

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

       
        mask = self.process_mask(mask)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        return img, mask.long()


def get_dataloaders(config):

    image_dir = [
        "C:/Users/aliqa/Desktop/Autonomous driving semantic segmentation/carla_dataset/dataA/dataA",
        "C:/Users/aliqa/Desktop/Autonomous driving semantic segmentation/carla_dataset/dataB/dataB",
        "C:/Users/aliqa/Desktop/Autonomous driving semantic segmentation/carla_dataset/dataC/dataC",
        "C:/Users/aliqa/Desktop/Autonomous driving semantic segmentation/carla_dataset/dataD/dataD",
        "C:/Users/aliqa/Desktop/Autonomous driving semantic segmentation/carla_dataset/dataE/dataE",
    ]

    data = CarlaDataset(img_dir=image_dir, transform=None)
    train_size = int(0.8 * data.__len__())
    test_size = data.__len__() - train_size
    train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])
    train_batch = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                             num_workers=config["num_workers"])
    test_batch = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"])

    return train_batch, test_batch