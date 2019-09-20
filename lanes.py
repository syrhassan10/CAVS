import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from ipywidgets import widgets
from IPython.display import display 
from IPython.display import Image
from ipywidgets import interactive, interact, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from skimage import img_as_ubyte
from thresholding_main import *
from calibration_main import *
from perspective_regionofint_main import *
from sliding_main import *
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#used for creating the image to write the final doc
writeup = 0
#used in interactive thresholding operation
verbose_threshold = 1

#Threshold operation for debuggin only
if writeup == 1:
    img = mpimg.imread("test_images/straight_lines2.jpg")
    mtx, dist = get_camera_calibration()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    if verbose_threshold == 1:
        interact (thresholding_interative, img=fixed(img), adp_thr = (0,255), k_size = (1,31,2), grad_thx_min =(0,255), 
              grad_thx_max =(0,255),
              grad_thy_min =(0,255), grad_thy_max = (0,255), mag_th_min = (0,255),
              mag_th_max = (0,255), dir_th_min  = (0,2,0.1), dir_th_max = (0,2,.1), 
              s_threshold_min = (0,255), 
              s_threshold_max = (0,255), v_threshold_min = (0,255), v_threshold_max = (0,255));


#Undistort for writeup
if writeup == 1:
    #test calibration for some image
    mtx, dist = get_camera_calibration()
    img = mpimg.imread("test_images/straight_lines2.jpg")
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(dst)
    ax2.set_title('Undistorted', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#Prespective transform for writeup
if writeup == 1:
    #img = mpimg.imread("test_images/test3.jpg")
    img = mpimg.imread("test_images/test6.jpg")

    top_down, M = perspective_transform(img)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    pts = np.array([[2, img.shape[0]-10], [img.shape[1]-5, img.shape[0]-10], [.55*img.shape[1], 0.625*img.shape[0]], [.45*img.shape[1], 0.625*img.shape[0]]], np.int32)
    #cv2.polylines(img, [pts], True, (0,255,255), 3)
    pts = np.array([[0.75*img.shape[1],5],[0.75*img.shape[1],img.shape[0]-5], [0.25*img.shape[1],img.shape[0]-5],[0.25*img.shape[1],5]], np.int32)
    #cv2.polylines(top_down, [pts], True, (0,255,255), 3)
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(top_down)
    ax2.set_title('Perspective transformed', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#region of interest for writeup:
#Undistort
if writeup == 1:
    #test calibration for some image
    mtx, dist = get_camera_calibration()
    img = mpimg.imread("test_images/straight_lines2.jpg")
    imshape = img.shape
    vertices = np.array([[(.55*imshape[1], 0.6*imshape[0]), (imshape[1],imshape[0]),
                        (0,imshape[0]),(.45*imshape[1], 0.6*imshape[0])]], dtype=np.int32)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    dst1 = region_of_interest(dst, vertices)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(dst)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(dst1)
    ax2.set_title('Region of Interest', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def draw_on_original(undist, left_fitx, right_fitx, ploty,Minv):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane with low confidence region in red
    cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))
    
    #confidence region in green
    shift = 50
    diff = (right_fitx - left_fitx)/2
    pts_left = np.array([np.transpose(np.vstack([left_fitx[400:], ploty[400:]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[400:], ploty[400:]])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)
    return resultv


mtx, dist = get_camera_calibration()

def pipeline(img):
    #to select whether diagnostic video(1) or submission video(0)
    verbose = 0
    #undistor the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #apply the thresholding operation
    thresh_combined, grad_th, col_th = thresholding(undist)
    #Perspective transformation
    perspective, Minv = perspective_transform(thresh_combined)
    perspective = cv2.cvtColor(perspective, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    #pass the perspective image to the lane fitting stage
    slides_pers, left_fitx, right_fitx, ploty, avg_cur, dist_centre_val = for_sliding_window(perspective)
    #draw the detected lanes on the original image 
    mapped_lane = draw_on_original(undist, left_fitx, right_fitx, ploty, Minv)
    #font and text for drawing the offset and curvature 
    curvature = "Estimated lane curvature %.2fm" % (avg_cur)
    dist_centre = "Estimated offset from lane center %.2fm" % (dist_centre_val)
    font = cv2.FONT_HERSHEY_COMPLEX
    # using cv2 for drawing text/images in diagnostic pipeline.
    if verbose == 1:
        middlepanel = np.zeros((120, 900, 3), dtype=np.uint8)
        l1 = np.zeros((50, 50, 3), dtype=np.uint8)
        l2 = np.zeros((50, 50, 3), dtype=np.uint8)
        l3 = np.zeros((50, 50, 3), dtype=np.uint8)
        l4 = np.zeros((50, 50, 3), dtype=np.uint8)
        l5 = np.zeros((50, 50, 3), dtype=np.uint8)
        l6 = np.zeros((50, 50, 3), dtype=np.uint8)
        l7 = np.zeros((50, 50, 3), dtype=np.uint8)
        legend = np.zeros((240, 1200, 3), dtype=np.uint8)

        cv2.putText(middlepanel, curvature, (30, 60), font, 1, (255,255,255), 2)
        cv2.putText(middlepanel, dist_centre, (30, 90), font, 1, (255,255,255), 2)
        cv2.putText(l1,"1", (15, 35), font, 1, (255,255,0), 2)
        cv2.putText(l2,"2", (15, 30), font, 1, (255,255,0), 2)
        cv2.putText(l3,"3", (15, 30), font, 1, (255,255,0), 2)
        cv2.putText(l4,"4", (15, 30), font, 1, (255,255,0), 2)
        cv2.putText(l5,"5", (15, 30), font, 1, (255,255,0), 2)
        cv2.putText(l6,"6", (15, 30), font, 1, (255,255,0), 2)
        cv2.putText(l7,"7", (15, 30), font, 1, (255,255,0), 2)
        text = "1-Detected Lanes, 2-Color Threshold\n3-Gradient Threshold, 4-Thresholding operations combined\n5-Perspective Transformation, 6-Original Frame\n7-Mapping Polynomials, Blue line-current frame polynomial fit,\nGreen line-smoothened polynomial fit, Pink - Lane pixels"

        y0, dy = 50, 40
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(legend, line, (50, y ), font, 1, (255,255,255),2)

        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        #2
        diagScreen[0:360, 1200:1560] = cv2.resize(np.dstack((col_th*255,col_th*255, col_th*255)), (360,360), interpolation=cv2.INTER_AREA) 
        #3
        diagScreen[0:360, 1560:1920] = cv2.resize(np.dstack((grad_th*255,grad_th*255,grad_th*255)), (360,360), interpolation=cv2.INTER_AREA) 
        #4
        diagScreen[360:720, 1200:1560] = cv2.resize(thresh_combined*255, (360,360), interpolation=cv2.INTER_AREA) 
        #5
        diagScreen[360:720,1560:1920] = cv2.resize(np.dstack((perspective*255, perspective*255, perspective*255)), (360,360), interpolation=cv2.INTER_AREA) 
        #7
        diagScreen[720:1080,1560:1920] = cv2.resize(slides_pers, (360,360), interpolation=cv2.INTER_AREA) 
        #6
        diagScreen[720:1080,1200:1560] = cv2.resize(img, (360,360), interpolation=cv2.INTER_AREA) 
        #1
        diagScreen[0:720, 0:1200] = cv2.resize(mapped_lane, (1200,720), interpolation=cv2.INTER_AREA) 

        #radii,offset and legend here
        diagScreen[720:840, 0:900] = middlepanel
        diagScreen[0:50, 0:50] = l1
        diagScreen[0:50, 1200: 1250] = l2
        diagScreen[0:50, 1560:1610] = l3
        diagScreen[720:770, 1560:1610] = l7
        diagScreen[360:410, 1560:1610] = l5
        diagScreen[720:770, 1200:1250] = l6
        diagScreen[360:410, 1200:1250] = l4
        diagScreen[840:1080, 0:1200] = legend
        #if diagnosis then return this image 
        return diagScreen
    #else return the original mapped imaged with the curvature and offset drawn
    cv2.putText(mapped_lane, curvature, (30, 60), font, 1.2, (255,0,0), 2)
    cv2.putText(mapped_lane, dist_centre, (30, 120), font, 1.2, (255,0,0), 2)
    return mapped_lane

while True:
    white_output = 'result.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    if cv2.waitKey(1)== ord('s'):
        break