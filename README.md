# Udacity-Lane-Detection-P1


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

To find the lanes I used below steps as my pipeline:-
* Convert Image to grayScale
* Use CannyEdge to find the edges
* Using OpenCV Morphological dilate operation to make edges more thick, because some edges were too much broken
* Apply the region of interest and keep only the edges which are present in Region of Interest
* Use hough algorithm to detect the lines from the image
* From these lines get the left and right lines using Slope and also exclude lines below a specific thershold(.2 in our case)
* Now using the points of left and right line get the m and c of y = mx + c by using np.polyfit and as we are using straight line we are using degree = 1
* Now we are getting m and c for left and right lane.
* As we know our region of interest and we know that value of y will be fixed in region of interest. We know y_min and y_max i.e. y1 and y2 of our lane line
* Now converting the eq of line from y = mx + c to x = (y-c)/m we take absolute value of x from here
* Now we have x1, y2 , x2, y2 using which we can draw lane lines
* As we can see that lanes are less thicker at top we are using fillpoly to draw and fill lines and we are setting the width at the base of polygon more that the width at the top of the image


### 2. Identify potential shortcomings with your current pipeline

* Lane lines are flickering.
* Lines are slightly off at the top for the broken lanes

### 3. Suggest possible improvements to your pipeline
* To work with optional challenge
* Reduce the flickering of the lanes
