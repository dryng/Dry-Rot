import numpy as np
from PIL import Image
from inference import ClassificationModel, SegmentationModel

segmentation = SegmentationModel()
classification = ClassificationModel("efficient_net_b3")

image = np.asarray(Image.open("exampleImg.png"))

# segmentation
segmentation_mask = segmentation.predict(image)
print(f"mask shape: {segmentation_mask.shape}")
print(f"transposed back into (H, W, C): {np.transpose(segmentation_mask, (2, 1, 0)).shape}")

# classification
classification_prediction = classification.predict(image)
print(f"classification prediction: {classification_prediction}")

