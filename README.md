# Real Time Object Tracking

**TODO: Add sample code**

## Overview
This project is a forked and modified version of the git repository by [kcg2015](https://github.com/kcg2015/Vehicle-Detection-and-Tracking). The repository has been simplified to track multiple objects in real time and organized into classes for easier usage in other packages. The tracking algorithm used here is a simple Kalman Filter.

![Real time video output from a webcam with the bounding boxes overlayed using the tracking algorithm described here](https://github.com/omarabid59/Object-Tracking-Library/blob/master/tracking_output.png)

## Quick Start
To incorporate the tracking into your own projects, we assume you have the output bounding boxes using a prebuilt model such as Faster R-CNN or SSD. See the [Tensorflow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for more details.

**Import and initialize the Tracker**
Assuming that you have the repository code in a folder called 'Tracking' and it is in your system path, import then create an instance. We can also specify the ```MAX_TRACKERS``` to indicate the maximum number of objects we can track at a time. There is no limit to the number of objects that can be tracked, however, it may impact performance if running in real time.
```
from Tracking.main_tracker import GlobalTracker
global_tracker = GlobalTracker(MAX_TRACKERS=15)
```

**Run the session. Output bounding boxes**
First, we need to run our model and output the bounding boxes as well as additional information. An example implementation can be found **coming soon**.
```
[bounding_boxes, scores, class_labels,num_detections] = sess.run(
              [boxes, scores, classes, num_detections],
               feed_dict={image_tensor: image_np_expanded})
```


**Feed the output into the tracking pipeline.**
The tracking pipeline automatically runs the Kalman filter and returns the updated set of bounding boxes.
```
[bounding_boxes, scores, class_labels,num_detections,
img] = global_tracker.pipeline(bounding_boxes,
                             scores,
                             class_labels,
                             image_np,
                             score_thresh=0.5)
```
**Display the tracking in real time.**
```
while True:
    if bounding_boxes != ():    
        output_frame = vis_util.visualize_boxes_and_labels_on_image_array(
          img,
          bounding_boxes,
          class_labels.astype(np.int32),
          scores,
          category_labels,
          use_normalized_coordinates=True,
          line_thickness=3,
          min_score_thresh=0.1)
    cv2.imshow('frame', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        done = True
        break
```                  
           

## Changes
- detector.py: Able to detect all classes above a threshold 'score_thresh' (default value of 0.3) defined in 'get_localization'.
          The get_localization function also returns an 'idx_vector' containing an array of indices for which the localization process is detected. This is useful for mapping the bounding boxes to it's corresponding classes and scores so it can be plotted on the image.
-renamed 'main.py' to 'main_tracker' and made it into a class named 'GlobalTracker()'. We can call this directly.
    - added a input argument to pass the maximum number of trackings that can be performed at any given time. with 'MAX_TRACKERS'
    - modified the 'pipeline' function to return an array of [_boxes, out_scores_arr, out_classes_arr, img]


