import cv2
import numpy as np
import argparse

def str2bool(string):
    str2val = {"True": True, "False": False, "false": False, "true": True}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def scale(img, scale=0.9):
    h,w = img.shape[:2]
    h = int(h*scale)
    w = int(w*scale)
    return cv2.resize(img, (w,h))

def parse_video(video_path, cut_off=40, show_video=True):
    """
    Go over the video, check the matches from one frame to another. Only accept frames that have less than cutoff features between them
    returns a list with all the images (loaded) that have passed the test
    """
    # open video
    cap = cv2.VideoCapture(video_path) # your video here
    counter = 0

    # init
    image_list = []    
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    _, last = cap.read()
    last = scale(last)
    prev = None
    image_list.append(last)
    kp1, des1 = orb.detectAndCompute(last, prev)

    # Just accepted images with less cutoff feature matches
    cutoff = 50     

    while True:
        # get frame
        ret, frame = cap.read()
        if not ret:
            break

        # resize
        frame = scale(frame)

        # count keypoints
        kp2, des2 = orb.detectAndCompute(frame, None)

        # match
        matches = bf.knnMatch(des1, des2, k=2)

        # lowe's ratio
        matched_points = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                matched_points.append(m)

        # check against cutoff
        if len(matched_points) < cutoff:
            # swap and save
            counter += 1
            last = frame
            des1 = des2
            image_list.append(last)

        # show
        if show_video:
            cv2.imshow("Frame(scaled)", frame)
            cv2.waitKey(1)
        prev = frame

    # also save last frame
    image_list.append(last)
    return image_list

def stitch_frames(image_list, out_file="panorama.png", show_output=True):
    """
    Given a list of images, it will try and stitch them together
    """

    # use built in stitcher
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(image_list)
    
    cv2.imwrite(out_file, stitched)

    if show_output:
        cv2.imshow("Stitched(scaled)", scale(stitched, 0.5))
        cv2.waitKey(0)

    return stitched


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="+", type=str, help="path to the file")
    parser.add_argument("--display", type=str2bool, default=True, help="whether to display output or not")
    parser.add_argument("--out_path", type=str, default="", help="path to output file")

    args = parser.parse_args().__dict__
    print(args)
    video_path: str = args.pop("video_path")
    out_path: str = args.pop("out_path")
    display: bool = args.pop("display")

    for video in video_path:
        video_name = video.split(".")[0]
        image_list = parse_video(video, show_video=display)

        if out_path == "":
            out_path = video_name + ".png"
        stitched = stitch_frames(image_list, out_file=out_path, show_output=display)
