# DA6401-Assignment2
Build and experiment with CNN based image classifiers using a subset of the iNaturalist dataset

The images are of different dimensions and the highest length and breadth of any image on both the train and test is 800. All the images will be resized to 256 x 256 through the default interpolation mode [(BILINEAR interpolation)](partA/dataloader.py#L11) and will become before being fed into the model.

If Augmentation is set to True, the training dataloader will be include RandomHorizontalFlip with 0.5 probability and RandomRotation by 15 degrees [augmentations](partA/dataloader.py#L17)