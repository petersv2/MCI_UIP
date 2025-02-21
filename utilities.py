import os
import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops_table



def get_mci_image(folder_path, core_name, sigma, percentage_removal):
    file_path_mask = os.path.join(folder_path,core_name,'mask', f'{core_name}.nhdr') # mask
    file_path_raw = os.path.join(folder_path,core_name,'MCI', f'{core_name}_MCI{sigma}.nhdr') # MCI image
    image_mask = sitk.ReadImage(file_path_mask)
    data_mask = sitk.GetArrayFromImage(image_mask)
    image_raw = sitk.ReadImage(file_path_raw)
    data_raw = sitk.GetArrayFromImage(image_raw)
    non_zero_slices = [i for i in range(data_mask.shape[0]) if np.any(data_mask[i])]


    start = non_zero_slices[0] #0
    end = non_zero_slices[-1] #-1
    image = data_mask*data_raw
    image = np.nan_to_num(image)
    image_non_zero = image[start:end,:,:]
    data_mask = data_mask[start:end,:,:]
    num_slices = image_non_zero.shape[0]
    slices_to_remove = int(np.ceil(num_slices)*(percentage_removal/100))
    image_mci_refined = image_non_zero[slices_to_remove:-slices_to_remove,:,:]
    data_mask_refined = data_mask[slices_to_remove:-slices_to_remove,:,:]

    return image_mci_refined,data_mask_refined

def get_CT_image(folder_path, core_name, percentage_removal):
    file_path_mask = os.path.join(folder_path,core_name,'mask', f'{core_name}.nhdr') # mask
    file_path_raw = os.path.join(folder_path,core_name,'CT', 'CT_image.nhdr') # CT image
    image_mask = sitk.ReadImage(file_path_mask)
    data_mask = sitk.GetArrayFromImage(image_mask)
    image_raw = sitk.ReadImage(file_path_raw)
    data_raw = sitk.GetArrayFromImage(image_raw)
    non_zero_slices = [i for i in range(data_mask.shape[0]) if np.any(data_mask[i])]
    start = non_zero_slices[0]
    end = non_zero_slices[-1]
    image = data_mask*data_raw
    image = np.nan_to_num(image)
    image_non_zero = image[start:end,:,:]
    data_mask = data_mask[start:end,:,:]
    num_slices = image_non_zero.shape[0]
    slices_to_remove = int(np.ceil(num_slices)*(percentage_removal/100))
    image_refined = image_non_zero[slices_to_remove:-slices_to_remove,:,:]
    data_mask_refined = data_mask[slices_to_remove:-slices_to_remove,:,:]

    return image_refined,data_mask_refined


def check_footprint(mask,nzt):

    min_row=10000
    min_col=10000
    max_row=0
    max_col=0

    num_slices = mask.shape[0]
                    
    for idx in range(num_slices):  
                 
        mask_slice = mask[idx,:,:]
        non_zero_content = mask_slice.sum()
        if non_zero_content > nzt:

            props = regionprops_table(mask_slice)
            dim1 = props['bbox-2'][0]-props['bbox-0'][0]
            dim2 = props['bbox-3'][0]-props['bbox-1'][0]

            if(props['bbox-0'][0] < min_row):
                min_row=props['bbox-0'][0]
            if(props['bbox-1'][0] < min_col):
                min_col=props['bbox-1'][0]
            if(props['bbox-2'][0] > max_row):
                max_row=props['bbox-2'][0]
            if(props['bbox-3'][0] > max_col):
                max_col=props['bbox-3'][0]

    return min_row,min_col,max_row,max_col