import cv2
import os
import multiprocessing
import numpy as np
import sys

sys.path.append('../inference')

from inference import SegmentationModel

def generate_patches(img_address,mask_address,width=256,height=256):

    annotated_patches = {}

    img = cv2.imread(img_address)
    mask = cv2.imread(mask_address)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

    if mask is None:
        return 

    x = 0
    y = 0
    i = 0

    mask = mask/255
    mask = mask.astype('uint8')

    while x+width<img.shape[1]:

        y = 0

        while y+height<img.shape[0]:
            sub_img = img[y:y+height,x:x+width]
            sub_mask = mask[y:y+height,x:x+width]

            annotated_patches[i] = {'patch':sub_img,'mask':sub_mask}

            i+=1
            y+=height

        x+=width

    return annotated_patches

def quantify_single_image(args):
    image_path = args[0]
    label_path = args[1]
    patches = generate_patches(image_path,label_path)
    
    segmentation = SegmentationModel(checkpoints_path="/work/dryngler/dry_rot/Dry-Rot/inference/")

    total = len(patches.keys())*256*256
    gt_dryrot_count = 0.0
    pr_dryrot_count = 0.0

    for i in patches:
        gt_mask = patches[i]['mask']
        segmentation_mask = segmentation.predict(patches[i]['patch']).squeeze().astype('uint8')
        gt_dryrot_count+=np.count_nonzero(gt_mask)
        pr_dryrot_count+=np.count_nonzero(segmentation_mask)
        
    gt_quantification = gt_dryrot_count/total
    pr_quantification = pr_dryrot_count/total

    print(gt_quantification,pr_quantification)
    
def quantify_all_images(path_patches,path_images,path_image_labels):
    files = os.listdir(path_patches)
    files = list(set([f.split('-')[0] for f in files]))
    
    list_args = []
    for file in files:
        list_args.append((os.path.join(path_images,f"{file}.JPG"),os.path.join(path_image_labels,f"{file}.png")))
        if len(list_args)>5:
            break
    
    with multiprocessing.Pool(int(multiprocessing.cpu_count()/2)) as pool:
        result = pool.map(quantify_single_image,list_args)
        
quantify_all_images('/space/ariyanzarei/dry_rot/datasets/2021-12-05_labeling/test/images','/space/ariyanzarei/dry_rot/raw_data/dry_rot_all_images','/space/ariyanzarei/dry_rot/raw_data/dry_rot_all_labels')