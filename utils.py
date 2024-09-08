import random
import torch
import numpy as np
import os
import cv2

from torch.utils.data import Dataset

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopper:
    def __init__(self, patience=30, min_delta=0):
        '''
        Arguments:
            patience (int): number of times to allow for no improvement before stopping the execution
            min_delta (float): minimum change counted as an improvement
        '''
        self.patience = patience 
        self.min_delta = min_delta
        self.counter = 0 # internal counter
        self.min_validation_loss = np.inf

    # Return True when validation loss is not decreased by `min_delta` `patience` times 
    def early_stop(self, validation_loss):
        if ((validation_loss + self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif ((validation_loss + self.min_delta) > self.min_validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def get_patience(self):
        return self.patience

# Modify the format of the dataset paths if required
class ImageDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data = os.path.join(root, 'DataSet', 'Train' if train else 'Validation')
        self.labels = np.load(os.path.join(root, 'Labels', f"{'Train' if train else 'Validation'}_labels.npy"))
        self.transform = transform
        self.ttv = "Train" if train else "Validation"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name =  self.ttv + "_" + str(idx) + ".jpg"
        image = cv2.imread(os.path.join(self.data, image_name))
        label = torch.tensor(self.labels[idx], dtype=torch.uint8)
        
        if self.transform:
            image = self.transform(image)
        return image, label