### Behavioral Cloning Project

This project requires the training of a convnet through Keras.  Images are produced in a vehicle simulator and trained to produce a steering angle based on forward-facing video input.

The car is tested on track one.  The car starts in autonomous mode on a relative straightaway.  The first left turn (Turn 1) leads to a bridge and then another left turn (Turn 2).  Next is a small straightaway followed by a right turn (Turn 3).  The last and final left turn (Turn 4) completes the lap.  

The students are issued a data set of 24108 images to train the network.  I augmented that with several more batches of images of 32895 and 6288 images.  The first batch of images included driving around the track smoothly with a controller.  I modified the throttle input to be the right analog stick to get an almost constant 20 MPH around the track.  The second batch of images were all "recovery" images of Turn 1, Turn 2 and Turn 3.
 
In pre-processing the images, I started with the raw image (320x160):

![](sample_raw.jpg)

Next I cropped off the sky and the hood of the car (320x95):

![](sample_crop.jpg)

Finally, I downsized the image 25% and converted from RGB to HSV (240x72):

![](sample_convert.jpg)

I gathered all the processed images in-memory.  The MacBook and AWS instance I used with this project have many GB of RAM, but the AWS instance only has 4GB of video memory.  For this reason, I used  

For my neural net, I started using the "VGG-like convnet" described [here](https://keras.io/getting-started/sequential-model-guide/) but could never get past Turn 1.  Then I looked around online at the NVIDIA documentation and at other students to see how they   

![](auto_mode.gif)
