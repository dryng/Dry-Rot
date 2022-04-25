from calendar import prmonth
from code import interact
from curses import savetty
import cv2
import os
import json
import numpy as np
import sys
from PIL import Image

sys.path.append('../inference')

from inference import SegmentationModel

def generate_patches(img_address,mask_address,width=256,height=256):

    annotated_patches = {}

    img = np.asarray(Image.open(img_address))
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

            annotated_patches[i] = {'patch':sub_img,'mask':sub_mask, 'row_start':y, 'row_end': y+height, 'col_start':x, 'col_end':x+width}

            i+=1
            y+=height

        x+=width

    return annotated_patches, img

def quantify_single_image(args):
    image_path = args[0]
    label_path = args[1]
    model = args[2]
    save_path = args[3]
    patches, img = generate_patches(image_path,label_path)
    
    total = len(patches.keys())*256*256
    gt_dryrot_count = 0.0
    pr_dryrot_count = 0.0

    GT_mask = np.zeros(img.shape)
    PR_mask = np.zeros(img.shape)

    for i in patches:
        gt_mask = patches[i]['mask']
        pred = model.predict(patches[i]['patch'])     
        segmentation_mask = pred.squeeze().astype('uint8')

        z = np.zeros(gt_mask.shape)
        
        if gt_mask.shape[0] != patches[i]['row_end'] - patches[i]['row_start'] or gt_mask.shape[1] != patches[i]['col_end'] - patches[i]['col_start']:
            continue

        GT_mask[patches[i]['row_start']:patches[i]['row_end'],patches[i]['col_start']:patches[i]['col_end'],:] = np.stack([z,gt_mask,z],axis=2)
        PR_mask[patches[i]['row_start']:patches[i]['row_end'],patches[i]['col_start']:patches[i]['col_end'],:] = np.stack([segmentation_mask,z,z],axis=2)
        
        gt_dryrot_count+=np.count_nonzero(gt_mask)
        pr_dryrot_count+=np.count_nonzero(segmentation_mask)
        
    alpha = 0.4


    Intersection = np.sum(np.logical_and(GT_mask[:,:,1],PR_mask[:,:,0]))
    Union = np.sum(np.logical_or(GT_mask[:,:,1],PR_mask[:,:,0]))
    IoU = Intersection/Union
    
    TP = np.count_nonzero(np.where((GT_mask[:,:,1]==1) & (PR_mask[:,:,0]==1)))
    FP = np.count_nonzero(np.where((GT_mask[:,:,1]==0) & (PR_mask[:,:,0]==1)))
    TN = np.count_nonzero(np.where((GT_mask[:,:,1]==0) & (PR_mask[:,:,0]==0)))
    FN = np.count_nonzero(np.where((GT_mask[:,:,1]==1) & (PR_mask[:,:,0]==0)))

    GT_mask = GT_mask*255
    GT_mask = GT_mask.astype('uint8')
    PR_mask = PR_mask*255
    PR_mask = PR_mask.astype('uint8')

    GT_mask = cv2.addWeighted(GT_mask, alpha, img, 1 - alpha, 0)
    PR_mask = cv2.addWeighted(PR_mask, alpha, img, 1 - alpha, 0)

    GT_mask = cv2.cvtColor(GT_mask,cv2.COLOR_RGB2BGR)
    PR_mask = cv2.cvtColor(PR_mask,cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(save_path,image_path.split('/')[-1].replace('.JPG','_GT.JPG')),GT_mask)
    cv2.imwrite(os.path.join(save_path,image_path.split('/')[-1].replace('.JPG','_PR.JPG')),PR_mask)

    gt_quantification = gt_dryrot_count/total
    pr_quantification = pr_dryrot_count/total

    print(f":: Img: {image_path.split('/')[-1]}\n\tGT: {gt_quantification}\n\tPR: {pr_quantification}\n\tIoU: {IoU}")
    sys.stdout.flush()

    return {"Img": image_path.split('/')[-1], "GT": gt_quantification, "PR": pr_quantification, "IOU":IoU, \
        "TP": TP, "FP": FP, "TN": TN, "FN": FN}
    
def quantify_all_images(path_patches,path_images,path_image_labels,save_path,model_name):
    files = os.listdir(path_patches)
    files = list(set([f.split('-')[0] for f in files]))
    
    model = SegmentationModel(checkpoints_path="/space/dryngler/dry_rot/experiments/22/checkpoints/DICE_epoch_88_unet_checkpoint.pth.tar")

    save_path = os.path.join(save_path,model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    list_args = []
    for file in files:
        # if file != "IMG_0333":
        #     continue
        list_args.append((os.path.join(path_images,f"{file}.JPG"),os.path.join(path_image_labels,f"{file}.png"),model,save_path))
        # if len(list_args)>5:
        #     break
    
    result = []
    for i,arg in enumerate(list_args):
        print("----------------------------------")
        print(f":: {i}/{len(list_args)}")
        result.append(quantify_single_image(arg))

    result_array = np.array([[r["GT"],r["PR"],r["IOU"],r["TP"],r["FP"],r["TN"],r["FN"]] for r in result])

    with open(os.path.join(save_path,"log.json"),'w+') as f:
        json.dump(result,f)

    TP = np.sum(result_array[:,3])
    FP = np.sum(result_array[:,4])
    TN = np.sum(result_array[:,5])
    FN = np.sum(result_array[:,6])
    
    IOU = TP/(TP+FN)
    Acc = (TP+TN)/(TP+FP+TN+FN)
    Precision = 1 if (TP+FP) == 0 else TP/(TP+FP)
    Recall = 1 if (TP+FN) == 0 else TP/(TP+FN) 
    F = 1 if (Precision+Recall) == 0 else 2*(Precision*Recall)/(Precision+Recall)

    print(f"Correlation between the Ground Truth and prediction quantification values: {np.corrcoef(result_array[:,0],result_array[:,1])[0,1]}")
    print(f"IoU between the Ground Truth and prediction masks: {IOU}")
    print(f"Accuracy (Pixlewise): {Acc}")
    print(f"Precision (Pixlewise): {Precision}")
    print(f"Recall (Pixlewise): {Recall}")
    print(f"F-1 (Pixlewise): {F}")
        
quantify_all_images('/space/ariyanzarei/dry_rot/datasets/2022-04-18_labeling/test/images',\
    '/space/ariyanzarei/dry_rot/raw_data/dry_rot_all_images',\
    '/space/ariyanzarei/dry_rot/raw_data/dry_rot_all_labels',\
    '/space/ariyanzarei/dry_rot/image_results/segmentation',\
    'unet')
