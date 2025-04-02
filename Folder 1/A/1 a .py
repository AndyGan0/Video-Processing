import cv2
import numpy as np
import json




def calculateResidualFrames(video):
    #   Gets a video and returns a list with the first frame and the rest of the frames as residual frames
    #   Note: the residual frames can't be displayed as image because they can have negative values


    #   making the list with the I Frame and the residual frames
    videoFrames = []

    #   Adding the I frame
    if (video.isOpened()):
        ret, previousFrame = video.read()  
        videoFrames.append(previousFrame)              
    else:
        return    
    
    #   Adding the residual frame for every P frame
    while ret:
        ret, CurrentFrame = video.read()
        if (not ret): break

        ResidualFrame = np.subtract(CurrentFrame,previousFrame)
        videoFrames.append(ResidualFrame)
        previousFrame = CurrentFrame

    return videoFrames






def run_Length_encode(Frame):




    EncodedFrame = []
    dimension_y = Frame.shape[0]
    dimension_x = Frame.shape[1]
    number_of_color_profiles = int(Frame.shape[2])


    
    for k in range(number_of_color_profiles):  # Iterate over color channels
        for i in range(dimension_y):  # For every row
            count = 1
            for j in range(1, dimension_x): 
                if (Frame[i][j][k] == Frame[i][j-1][k]):
                    count += 1
                else:                    
                    EncodedFrame.append((Frame[i][j-1][k], count))
                    count = 1
            EncodedFrame.append((Frame[i][dimension_x - 1][k], count))



    return EncodedFrame






def run_Length_decode(EncodedFrame,Shape):


    decoded_frame = np.zeros(Shape, dtype=np.uint8)

    j = 0
    k = 0
    i = 0
    for pixel, count in EncodedFrame:

        for x in range(count):
            decoded_frame[i][j][k] = pixel        
            j += 1

            if (j>=Shape[1]):
                i += 1
                j = 0
                if (i>=Shape[0]):
                    i = 0
                    j = 0
                    k += 1
                    if k >= Shape[2]:
                        # Handle the case where k exceeds the number of color channels
                        return decoded_frame

    return decoded_frame


                

                



                










    

def encode(filename):
    #   Gets the file name of a video that is in the same folder as the encoder.py file
    #   Produces the residual frames and encodes the file using huffman technique
    #   the output file is exported in the same directory
    

    # getting video information
    video = cv2.VideoCapture(filename)

    framerate = video.get(cv2.CAP_PROP_FPS)

    videoFrames = calculateResidualFrames(video)





    shape = videoFrames[0].shape

    for frame in videoFrames:
        cv2.imshow('Residual Frames', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    
    EncodedFrames = []
    for frame in videoFrames:
        EncodedFrames.append(run_Length_encode(frame))
    

    


    return EncodedFrames,shape


    







def decode(EncodedVideo,Shape):


    VideoFrames = []
    for i in range(len(EncodedVideo)):    
        VideoFrames.append( run_Length_decode(EncodedVideo,Shape) )



    for i in range(1,len(VideoFrames)):
        VideoFrames[i] = np.add(VideoFrames[i-1],VideoFrames[i])


    
    for frame in VideoFrames:
        cv2.imshow('Decoded Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break






EncodedVideo,Shape = encode("video.mp4")

decode(EncodedVideo,Shape)



'''


cv2.imshow('Image', ResidualFrame)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''