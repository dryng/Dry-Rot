import numpy as np
from PIL import Image
from inference import ClassificationModel, SegmentationModel
import cv2

segmentation = SegmentationModel(checkpoints_path="/work/dryngler/dry_rot/Dry-Rot/inference/")
classification = ClassificationModel("efficient_net_b3",checkpoints_path="/work/dryngler/dry_rot/Dry-Rot/inference/")

image = np.asarray(Image.open("exampleImg.png"))

# segmentation
segmentation_mask = segmentation.predict(image)*255
print(segmentation_mask)
print(f"mask shape: {segmentation_mask.shape}")
print(f"transposed back into (H, W, C): {np.transpose(segmentation_mask, (2, 1, 0)).shape}")
cv2.imwrite("mask.png",segmentation_mask.astype('uint8').squeeze())

# classification
classification_prediction = classification.predict(image)
print(f"classification prediction: {classification_prediction}")

