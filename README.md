# Binary-Segmentation
An user interactive semi-automatic binary segmentation model implemented in OpenCV 3.3.0 and Python 2.7.
Given sparse markings of foreground and background by the user, it calculates SLIC superpixels, and runs a graph-cut algorithm.
Color histograms are calculated for all superpixels and foreground background. This algorithm takes into account superpixel-superpixel and superpixel-Foreground/Background interaction to obtain a final binary image segmentation.

Uploaded Files Description:


main-bonus.py: Python code to implement interactive semi-automatic binary segmentation of an image.


astronaut.png: Input image of an astronaut


astronaut-marking.png: Static foreground/background given as an input image.


mask.png: Resultant image of static markings.


Segmentation-output.gif: GIF result of user-interactive foreground/background markings.
