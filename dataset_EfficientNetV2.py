import torch
import numpy as np
from torchvision import transforms
import os
from utilities import get_CT_image, get_mci_image,check_footprint
from math import ceil

# Prior to invoking this code for loading image data, it is assumed that preprocessing as described in the
# paper has already been done. Then, when loading these preprocessed images, the following steps are taken. 
# Some of these steps are implemented the present file, some are implemented in the get_CT_image(), 
# get_mci_image() and check_footprint() functions in utilities.py
#
#  1. the top and bottom 10% of axial slices in the 3D CT scan are discared
#  2. In each remaining axial slice, the lungs are masked and a rectangular bounding box around 
#     the masked lungs is found
#  3. the axial slice is cropped around this bounding box
#  4. this cropped region is placed at the center of a 384x384 image filled with 0's outside the cropped region
#  5. the vast majority of slices in our data have a bounding box smaller than 384x384, except for a small 
#     number of slices where the larger dimension of the bounding box exceeds 384 by a few pixels. 
#     In such cases, the bounding box is trimmed on both ends of this dimension by an equal number of pixels 
#     on each side down to 384. 

class Dataset_EfficientNetV2:
    def __init__(self, folder_path, subject_meta, is_ct_image=False, is_shape=False,mci_sigma = 2, max_w_size = 384, percentage_removal=10,transform=None):
        self.folder_path = folder_path
        self.images = []
        self.masks = []
        self.labels = []
        self.cores = []
        self.transform = transform
        
        subject_meta.set_index('name',inplace=True)

        # List of all patient folder names
        folder_names = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

        for idx, folder in enumerate(folder_names):
                if(folder in subject_meta.index): 
                    print(f"Loading index, folder = {idx}, {folder}")   

                    if is_ct_image:
                        image_3d, mask = get_CT_image(folder_path, folder, percentage_removal)
                    else:
                        image_3d, mask = get_mci_image(folder_path, folder, mci_sigma, percentage_removal)

                    min_row, min_col, max_row, max_col = check_footprint(mask,100)
                    width = max_col-min_col
                    height = max_row-min_row

                    if width > max_w_size:
                        difw = width - max_w_size
                        width = max_w_size
                        trimw = ceil(difw/2)
                        min_col = min_col+trimw
                        max_col = max_col-trimw
                    if height > max_w_size:
                        difh = height-max_w_size
                        height = max_w_size
                        trimh = ceil(difh/2)
                        min_row = min_row+trimh
                        max_row = max_row-trimh

                    num_slices = image_3d.shape[0]

                    if is_shape:
                        image_3d=mask
                    
                    folder_label = subject_meta[subject_meta.index == folder]['Diagnosis'].values[0]
                    for idx in range(num_slices):                    
                        mask_slice = mask[idx,:,:]
                        image_slice = image_3d[idx,:,:]

                        mask_slice_crop = mask_slice[min_row:min_row+height,min_col:min_col+width]
                        image_slice_crop = image_slice[min_row:min_row+height,min_col:min_col+width]
                        padding = int((max_w_size-width)/2)
                        paddingH = int((max_w_size-height)/2)
                        mask_slice_pad = np.zeros((max_w_size,max_w_size))
                        image_slice_pad = np.zeros((max_w_size,max_w_size))
                        mask_slice_pad[paddingH:height+paddingH, padding:width+padding] = mask_slice_crop
                        image_slice_pad[paddingH:height+paddingH, padding:width+padding] = image_slice_crop
                        self.images.append(image_slice_pad.astype(np.float32))
                        self.masks.append(mask_slice_pad.astype(np.uint8))
                        self.labels.append(folder_label)
                        self.cores.append(folder)
                

    # Method to get a specific item
    def __getitem__(self, index):             
        image = self.images[index]
        mask = self.masks[index]
        label = self.labels[index]
        core = self.cores[index]

        # Apply transformations
        if self.transform:
            image = self.transform(image).astype(np.float32)
            mask = self.transform(mask).astype(np.uint8)
        else:
            # Default transform: Convert to tensor and unsqueeze
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
            mask = to_tensor(mask)

        image = image.to(torch.float32)

        return image, label, mask, core

    # Method to get the dataset size
    def __len__(self):
        return len(self.images)
