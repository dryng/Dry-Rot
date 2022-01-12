import numpy as np
import cv2
from PIL import Image
from inference import ClassificationModel, SegmentationModel
import cv2

segmentation = SegmentationModel(checkpoints_path="/work/dryngler/dry_rot/Dry-Rot/inference/")
classification = ClassificationModel("efficient_net_b3",checkpoints_path="/work/dryngler/dry_rot/Dry-Rot/inference/")

image = np.asarray(Image.open("exampleImg4.png"))

# segmentation
segmentation_mask = segmentation.predict(image)*255
print(segmentation_mask)
print(f"mask shape: {segmentation_mask.shape}")
cv2.imwrite('exampleMask4.jpg', segmentation_mask*255)
# pillow is a little more complicated to save but it still works
# Image.fromarray((segmentation_mask.squeeze(-1)*255).astype('uint8')).save("exampleMask.jpg") 

# classification
classification_prediction = classification.predict(image)
print(f"classification prediction: {classification_prediction}")

