import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, dataset  
from torch.utils.data import RandomSampler
import torch.nn.functional as F 
import numpy as np
from torchvision.transforms.transforms import Resize 
from data_path import train_imgs, valid_imgs, test_imgs, class_to_int

def get_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

class BirdDataset(Dataset):
    def __init__(self, imgs_list, class_to_int, transforms = None):
        super().__init__()
        self.imgs_list = imgs_list
        self.class_to_int = class_to_int
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.imgs_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        label = image_path.split("/")[-2]
        label = self.class_to_int[label]
        if self.transforms:
            image = self.transforms(image)
        return image, label
    
    def __len__(self):
        return len(self.imgs_list)


train_dataset = BirdDataset(train_imgs, class_to_int, get_transform())
valid_dataset = BirdDataset(valid_imgs, class_to_int, get_transform())
test_dataset  = BirdDataset(test_imgs, class_to_int, get_transform())

train_random_sampler = RandomSampler(train_dataset)
test_random_sampler = RandomSampler(test_dataset)
val_random_sampler = RandomSampler(valid_dataset)

train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    sampler=train_random_sampler,
    num_workers=4,
)

test_data_loader = DataLoader(
    dataset = test_dataset,
    batch_size=16,
    sampler=test_random_sampler,
    num_workers=4,
)

val_data_loader = DataLoader(
    dataset= valid_dataset,
    batch_size=16,
    sampler=val_random_sampler,
    num_workers=4,
)
