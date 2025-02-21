import os
import numpy as np
from torchvision import transforms

class DatasetSOFIA:
    def __init__(self, folder_path, subject_meta, is_ct_image=False, is_shape=False, mci_sigma=2,transform=None):
        self.folder_path = folder_path
        self.images = []
        self.labels = []
        self.cores = []
        self.transform = transform

        subject_meta.set_index('name',inplace=True)
        
        # List of all patient folder names
        folder_names = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

        # For each folder, load montages previously created by running create_SOFIA_montages.ipynb
        for idx, folder in enumerate(folder_names):

            if(folder in subject_meta.index): 
                print(f"Loading index, folder = {idx}, {folder}") 

                stack_output_path = os.path.join(folder_path,folder,'SOFIAmontages')
                if(is_ct_image):
                    images_path = os.path.join(stack_output_path, 'montage.npy')
                else:                
                    images_path = os.path.join(stack_output_path, f'montage_mci_{mci_sigma}.npy')
                
                folder_label = subject_meta[subject_meta.index == folder]['Diagnosis'].values[0]
                if os.path.exists(images_path):
                    images = np.load(images_path)    

                    if(is_shape):
                        images[images!=0]=1
                                
                    self.images.extend(images)
                    self.labels.extend([folder_label] * len(images)) 
                    self.cores.extend([folder]*len(images))

                else:
                    print(f"Montage not found in {folder}")

    # Method to get a specific item
    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]
        core = self.cores[index]

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: Convert to tensor and unsqueeze
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, label, core

    # Method to get the dataset size
    def __len__(self):
        return len(self.images)

