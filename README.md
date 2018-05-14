# Tracking for pre-detected objects

## Overview
This project is a forked and modified version of the git repository by [kcg2015](https://github.com/kcg2015/Vehicle-Detection-and-Tracking). The repository has been simplified to track multiple objects and organized into classes for easier usage in other packages. The tracking algorithm used here is a simple Kalman Filter.



## Quick Start
To incorporate the tracking into your own projects, we assume you have the output bounding boxes using a prebuilt model such as Faster R-CNN or SSD. See the [Tensorflow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for more details.




## Pipeline

We include two important design parameters, ```min_hits``` and ```max_age```, in the pipe line.  The parameter ```min_hits``` is the number of consecutive matches needed to establish a track. The parameter ```max_age``` is number of consecutive unmatched detection before a track is deleted. Both parameters need to be tuned to improve the tracking and detection performance.

The pipeline deals with matched detection, unmatched detection, and unmatched trackers sequentially. We annotate the tracks that meet the ```min_hits``` and ```max_age``` condition. Proper book keep is also needed to deleted the stale tracks. 

The following examples shows the process of the pipeline. When the car is first detected in the first video frame, running the following line of code returns an empty list , an one-element list, and an empty list  for ```matched```, ```unmatched_dets```, and ```unmatched_trks```, respectively. 

```
matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3) 
```
We thus have a situation of unmatched detections. Unmatched detections are processed by the following code block:

```
if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)
```
This code block carries out two important tasks, 1) creating a new tracker ```tmp_trk``` for the detection; 2) carrying out the Kalman filter's predict stage ```tmp_trk.predict_only()```. Note that this newly created track is still in probation period, i.e., ```trk.hits =0```, so this track is yet established after the end pipeline. The output image is the same as the input image - the detection bounding box is not annotated.
<img src="example_imgs/frame_01_det_track.png" alt="Drawing" style="width: 150px;"/>

When the car is  detected again in the second video frame, running the following ```assign_detections_to_trackers``` returns an one-element list , an empty list, and an empty list for matched, unmatched_dets, and unmatched_trks, respectively. As shown in the following figure, we have a matched detection, which will be processed by the following code block:

```
if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
```
This code block carries out two important tasks, 1) carrying out the Kalman filter's prediction and update stages ```tmp_trk.kalman_filter()```; 2) increasing the hits of the track by one ```tmp_trk.hits +=1```. With this update,  
the condition ```if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)) ``` is statified, so the track is fully established. As the result, the bounding box is annotated in the output image, as shown in the figure below.
<img src="example_imgs/frame_02_det_track.png" alt="Drawing" style="width: 150px;"/>
## Issues

The main issue is occlusion. For example, when one car is passing another car, the two cars can be very close to each other. This can fool the detector to output a single(and bigger bounding) box, instead of two separate bounding boxes. In addition, the tracking algorithm may treat this detection as a new detection and set up a new track.  The tracking algorithm may fail again when one the passing car moves away from another car. 

## Changes
- detector.py: Able to detect all classes above a threshold 'score_thresh' (default value of 0.3) defined in 'get_localization'.
          The get_localization function also returns an 'idx_vector' containing an array of indices for which the localization process is detected. This is useful for mapping the bounding boxes to it's corresponding classes and scores so it can be plotted on the image.
-renamed 'main.py' to 'main_tracker' and made it into a class named 'GlobalTracker()'. We can call this directly.
    - added a input argument to pass the maximum number of trackings that can be performed at any given time. with 'MAX_TRACKERS'
    - modified the 'pipeline' function to return an array of [_boxes, out_scores_arr, out_classes_arr, img]


