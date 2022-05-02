Experiments 

1)  name: check dice threshold 
    date: 2/20/22
    model: UNET
    description: run through validation set and save models with low threshold to see what is causing it (dirt? etc)

2)  name: clean build
    date: 2/26/22
    model: UNET
    description: fresh build of model. Trained well. Epoch was set to 40 and went through all 40 with more to go. 
    Set for more epochs 

3)  name: clean build #2
    date: 2/27/22
    model: UNET
    description: clean build above didn't save checkpoints so need to retrain.

4)  name: 4
    date: 2/27/22
    model: UNET
    description: clean build for 100 epochs

5)  name: 5
    date: 3/30/22
    model: UNET
    description: train segmentation model with (small) newly labeled images

6)  name: 6
    date: 3/30/22
    model: UNET
    description: LR exploration to see if theres improvement train segmentation model with (small) newly labeled images. 

7)  name: 7
    date: 3/30/22
    model: efficient net b3
    description: train classification model with (small) newly labeled images. 

8)  name: 8
    date: 3/30/22
    model: UNET
    description: train segmenation model with (small) newly labeled images for 100 epochs with early stopping. 

9)  name: 9
    date: 4/4/22
    model: UNET
    description: train segmenation model with bad dataset to see difference. 

10) name: 10
    date: 4/18/22
    model: UNET
    description: 100 epoch train on newly cleaned dataset. Goal of above 90%

11) name: 11
    date: 4/18/22
    model: resnet_18
    description: Classification train on new dataset. Goal of above 90%

12) name: 12
    date: 4/18/22
    model: mobilenet_v3_small
    description: Classification train on new dataset. Goal of above 90%

13) name: 13
    date: 4/18/22
    model: custom_mobilenet_v3_small
    description: Classification train on new dataset. Goal of above 90%

14) name: 14
    date: 4/18/22
    model: mobilenet_v3_large
    description: Classification train on new dataset. Goal of above 90%

15) name: 15
    date: 4/18/22
    model: efficient_net_b3
    description: Classification train on new dataset. Goal of above 90%

16) name: 16
    date: 4/18/22
    model: efficient_net_b4
    description: Classification train on new dataset. Goal of above 90%

22) name: 22
    date: 4/21/22
    model: UNET
    description: Unet trained for 88 epochs

23) name: 23
    date: 4/21/22
    model: UNET
    description: Unet with SGD

24) name: 24
    date: 4/27/22
    model: UNET
    description: train for 200 epochs with patience of 200

25) name: 25
    date: 4/27/22
    model: UNET
    description: train for 200 epochs with patience of 200. Batch size of 16

26) name: 26
    date: 4/27/22
    model: resnet_18
    description: train for 200 epochs with patience of 200

27) name: 27
    date: 4/27/22
    model: mobilenet_v3_small
    description: train for 200 epochs with patience of 200

28) name: 28
    date: 4/27/22
    model: custom_mobilenet_v3_small
    description: train for 200 epochs with patience of 200

29) name: 29
    date: 4/27/22
    model: mobilenet_v3_large
    description: train for 200 epochs with patience of 200

30) name: 30
    date: 4/27/22
    model: efficient_net_b3
    description: train for 200 epochs with patience of 200

31) name: 31
    date: 4/27/22
    model: efficient_net_b4
    description: train for 200 epochs with patience of 200

