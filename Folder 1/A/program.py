import cv2
import numpy as np



video = cv2.VideoCapture("video.mp4")


# getting video information
filename = "output.mp4"

#   fourcc = video.get(cv2.CAP_PROP_FOURCC)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

framerate = video.get(cv2.CAP_PROP_FPS)
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
resolution = (int(width), int(height))

# creating the output with residual frames
videoOutput = cv2.VideoWriter( filename, fourcc, framerate, resolution)



if (video.isOpened()):
    ret, previousFrame = video.read()
    videoOutput.write(previousFrame)
else:
    ret = False



while ret:
    ret, CurrentFrame = video.read()
    if (not ret): break

    videoOutput.write(CurrentFrame)

    previousFrame = CurrentFrame



    

    


video.release()
videoOutput.release()
cv2.destroyAllWindows()




