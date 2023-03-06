import numpy as np
import cv2 as cv
import argparse


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                    qualityLevel = 0.1,
                    minDistance = 7,
                    blockSize = 7 )
                    
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# Initiate SIFT and ORB detector
orb = cv.ORB_create()
sift = cv.SIFT_create()

def plot_everything(image_warped,frame,template_plot,out):
    image_warped = cv.resize(image_warped,(template_plot.shape[1],template_plot.shape[0]),
                                interpolation=cv.INTER_CUBIC)
    frame = cv.resize(frame,(frame.shape[1]//3,frame.shape[0]//3),
                            interpolation=cv.INTER_CUBIC)
    out.write(image_warped)
    cv.imshow("image_warped",image_warped)
    cv.imshow("frame",frame)
    cv.imshow("template",template_plot)

def find_matrix_betwen_images_orb(image,keypoints_template, descriptors_template):
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    keypoints_image, descriptors_image = orb.detectAndCompute(image,None)
    # compute the descriptors with ORB
    matches = bf.knnMatch(descriptors_image,descriptors_template,2)

    good_points=[] 

    for m, n in matches: 
        if(m.distance < 0.9*n.distance): 
            good_points.append(m) 
    if len(good_points)<6:
        print("Not enough points")
        return None
    # Extract the matched feature points
    points1 = np.float32([keypoints_image[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
    points2 = np.float32([keypoints_template[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)
    
    final_matrix, mask = cv.findHomography(points1, points2, cv.RANSAC,5)

    return final_matrix

def find_matrix_betwen_images_sift(gray_image,keypoints_template, descriptors_template):
    keypoints_image, descriptors_image = sift.detectAndCompute(gray_image,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    
    matches = bf.knnMatch(descriptors_image,descriptors_template,k=2)

    good_points=[] 
    
    for m, n in matches: 
        if(m.distance < 0.7*n.distance): 
            good_points.append(m) 
            
    if len(good_points)<6:
        print("Entrou")
        return None
    # Extract the matched feature points
    points1 = np.float32([keypoints_image[m.queryIdx].pt for m in good_points])
    points2 = np.float32([keypoints_template[m.trainIdx].pt for m in good_points])
    
    final_matrix, mask = cv.findHomography(points1, points2, cv.RANSAC,5)
    return final_matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="path to video")
    parser.add_argument("template_path",  type=str, help="path to template")
    parser.add_argument("--out_path", type=str, default="", help="path to output file")

    args = parser.parse_args().__dict__
    print(args)
    video_path: str = args.pop("video_path")
    template_path: str = args.pop("template_path")
    output_path: str = args.pop("out_path")

    #Load template
    template = cv.imread(template_path)
    template_grey = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    #Load Video and prepares_output
    cap = cv.VideoCapture(video_path)
    if output_path=="":
        video_name = video_path.split(".")[0]
        output_path=video_name + "_output.avi"
    out = cv.VideoWriter(output_path,cv.VideoWriter_fourcc('M','J','P','G'), 10, (template.shape[1]//3,template.shape[0]//3))

    #Pre-compute template orb and SIFT features
    keypoints_template_orb, descriptors_template_orb = orb.detectAndCompute(template,None)
    keypoints_template, descriptors_template = sift.detectAndCompute(template_grey,None)
    
    #Finds homography between first possible frame and template
    final_matrix=None
    while final_matrix is None:
        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_frame = cv.flip(old_frame, 0)
        old_frame=cv.flip(old_frame, 1)
        initial_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        initial_matrix = find_matrix_betwen_images_sift(initial_gray, keypoints_template, descriptors_template)

    template_plot = cv.resize(template,(template.shape[1]//4,template.shape[0]//4),
                                    interpolation=cv.INTER_CUBIC)
    
    #Find good ShiTomasi features in first frame to apply lucas kanade to it with the second frame
    p0 = cv.goodFeaturesToTrack(initial_gray, mask = None, **feature_params)
    n_big_movements = 0
    itteration = 0
    do_orb=False
    wait=0
    itterations_since_Orb=0
    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        #Flips horizontally and vertically becasue it wasa recorded on phone
        frame = cv.flip(frame, 0)
        frame=cv.flip(frame, 1)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray_warped = cv.warpPerspective(frame,initial_matrix,(template.shape[1],template.shape[0]))
        #Uses lucas kanade to track ShiTomasi features on previous frame
        p1, st, err = cv.calcOpticalFlowPyrLK(initial_gray, frame_gray_warped, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        wait=wait-1
        if good_new is not None and len(good_new)>3 and good_old is not None and len(good_old)>3:
            #Uses the good tracked features from both frames to estimate homography
            movement, mask = cv.findHomography(good_new, good_old, cv.RANSAC,5)
            #print(np.linalg.norm(movement))
            #If homography represents a big change it means its probably an error 
            if np.linalg.norm(movement) > 100:
                n_big_movements= n_big_movements+1
                #If there has been 3 big movements it uses orb to re-center
                if n_big_movements>2:          
                    do_orb=True
                    #Assumes current frames are out of focus so it waits a bit 
                    wait=4
                    n_big_movements=0
                continue
            if do_orb and wait==0:
                possible_matrix=find_matrix_betwen_images_orb(frame_gray,keypoints_template_orb,descriptors_template_orb)
                itterations_since_Orb=0
                #If it couldnt find enough points it waits and re-does orb in wait frames
                if possible_matrix is not None:
                    do_orb=False
                    print(np.linalg.norm(possible_matrix-final_matrix))
                    final_matrix=possible_matrix
                    movement=np.eye(3)
                else:
                    wait=4
            #Just multiply the matrix until this frame with the estimated
            # "movement" from this to the previous one        
            final_matrix = movement@initial_matrix
            image_warped = cv.warpPerspective(frame,final_matrix,(template.shape[1],template.shape[0]))
            #If the video takes more than 300 itterations without orb
            # it assumes the error is getting bigger so it redoes it
            # if itterations_since_Orb ==200:
            #     do_orb=True
            #     wait=1
            #     itterations_since_Orb=0
        #plots current frame, frame warped and template
        
        plot_everything(image_warped,frame,template_plot,out)
        itteration=itteration+1
        itterations_since_Orb = itterations_since_Orb+1
        k = cv.waitKey(1) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv.destroyAllWindows()