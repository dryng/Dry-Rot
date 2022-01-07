import cv2
import os
import multiprocessing
import numpy as np
import sys

sys.path.append('../inference')

from inference import ClassificationModel

def generate_patches(img_address,mask_address,width=256,height=256):

    annotated_patches = {}

    img = cv2.imread(img_address)
    mask = cv2.imread(mask_address)

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

def mask_contains_dryrot(mask):
    return np.max(mask) == 1

def quantify_single_image(args):
    image_path = args[0]
    label_path = args[1]
    patches = generate_patches(image_path,label_path)
    
    classification = ClassificationModel("efficient_net_b3",checkpoints_path="/work/dryngler/dry_rot/Dry-Rot/inference/")

    total = len(patches.keys())
    gt_dryrot_count = 0.0
    pr_dryrot_count = 0.0

    for i in patches:
        if mask_contains_dryrot(patches[i]['mask']):
            gt_dryrot_count+=1
        
        classification_prediction = classification.predict(patches[i]['patch'])
        if classification_prediction == 1:
            pr_dryrot_count+=1

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