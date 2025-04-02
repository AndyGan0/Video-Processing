import math
import cv2
import numpy as np



def calculateLastLevelMotionVectors(frame1, frame2, Highest_Level = 4, BlockSize = 64, Radius_k = 32):


    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    originalFrameAxis_x = frame1.shape[1]
    originalFrameAxis_y = frame1.shape[0]
    step = 2 ** (Highest_Level-1)


    #   making the motion vector 
    size_motion_vector_i = math.ceil(originalFrameAxis_y/BlockSize)
    size_motion_vector_j = math.ceil(originalFrameAxis_x/BlockSize)    
    MotionVector = np.zeros( (size_motion_vector_i , size_motion_vector_j,2) ,dtype = int )

    
   
    #   itterate through all the macroblocks
    for macroblock_i in range( size_motion_vector_i ):

        #   define the y axis of the macroblock
        target_block_i = macroblock_i * BlockSize
        target_block_size_i = BlockSize
        if (macroblock_i == size_motion_vector_i - 1):
            #   we are on the last row with macroblocks. the size may differ
            target_block_size_i = originalFrameAxis_y - target_block_i            

        for macroblock_j in range( size_motion_vector_j ):

            #   define the y axis of the macroblock      
            target_block_j = macroblock_j * BlockSize
            target_block_size_j = BlockSize
            if (macroblock_j == size_motion_vector_j - 1):
                #   we are on the last column with macroblocks. the size may differ
                target_block_size_j = originalFrameAxis_x - target_block_j


            #   match every macroblock with the best reference block

            bestScore = 256
            bestScoreMacroBlock = []


            #   iterrate through all possible reference blocks in the range of the radius k
            reference_block_i = target_block_i - Radius_k
            if reference_block_i < 0: reference_block_i = 0
            while ( reference_block_i + target_block_size_i <= originalFrameAxis_y and reference_block_i <= target_block_i + Radius_k ):  #    make sure the radius is inside the image and not out of bounds

                reference_block_j = target_block_j - Radius_k
                if reference_block_j < 0: reference_block_j = 0
                while ( reference_block_j + target_block_size_j <= originalFrameAxis_x and reference_block_j <= target_block_j + Radius_k ):
                    #   for every reference block in the radius, check the score

                    



                    Score = 0
                    Count = 0

                    for i in range( 0 , target_block_size_i, step ):    #   because of the higher level, the steps will be more. We dont compare all pixels of the macroblock
                        for j in range ( 0 , target_block_size_j , step ):
                            Score += abs( int(frame2[ reference_block_i + i ][ reference_block_j + j ]) - int(frame1[ target_block_i + i][ target_block_j + j]) )
                            Count += 1
                    
                    Score = Score / Count

                    if ( Score < bestScore ):
                        bestScore = Score
                        bestScoreMacroBlock = [reference_block_i,reference_block_j]


                    #   because of the higher level, the steps will be more. We dont compare all the possible reference blocks.
                    #   We compare only those accesible by this level (using the step)
                    reference_block_j += step
                
                reference_block_i += step

            MotionVector[macroblock_i][macroblock_j] = bestScoreMacroBlock
    

    return MotionVector





def calculateMotionVector(frame1, frame2, Highest_Level = 4, BlockSize = 64, Radius_k = 32):


    

    originalFrameAxis_y = frame1.shape[0]
    originalFrameAxis_x = frame1.shape[1]

    HigherLevelMotionVector = calculateLastLevelMotionVectors(frame1, frame2, Highest_Level, BlockSize, Radius_k)
    size_of_motion_vector_i = HigherLevelMotionVector.shape[0]
    size_of_motion_vector_j = HigherLevelMotionVector.shape[1]

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    MotionVector = HigherLevelMotionVector
    
    
    for level in range( Highest_Level-1 , 0 , -1 ):
        #   calculating Motion Vector for each level
        

        #   define the step
        step =  2 ** (level-1)

        for macroblock_i in range( size_of_motion_vector_i ):

            target_block_i = macroblock_i * BlockSize
            target_block_size_i = BlockSize
            if (macroblock_i == size_of_motion_vector_i - 1):
                #   we are on the last row with macroblocks. the size may differ
                target_block_size_i = originalFrameAxis_y - target_block_i 

            for macroblock_j in range( size_of_motion_vector_j ):

                target_block_j = macroblock_j * BlockSize
                target_block_size_j = BlockSize
                if (macroblock_j == size_of_motion_vector_j - 1):
                    #   we are on the last row with macroblocks. the size may differ
                    target_block_size_j = originalFrameAxis_x - target_block_j 

                
                #   we have defined the macroblock, time to find the best reference block
                bestScore = 256
                bestScoreMacroBlock = []

                
                PreviousLevelReferenceBlock_i = MotionVector[macroblock_i][macroblock_j][0]
                PreviousLevelReferenceBlock_j = MotionVector[macroblock_i][macroblock_j][1]



                #   iterrate through all reference blocks in the range of the radius 1
                for reference_block_i in range( PreviousLevelReferenceBlock_i-step , PreviousLevelReferenceBlock_i+step+1 , step ):                    
                    if reference_block_i < 0 or reference_block_i + target_block_size_i > originalFrameAxis_y : continue   #    make sure the reference block is inside the image and not out of bounds

                    for reference_block_j in range( PreviousLevelReferenceBlock_j-step , PreviousLevelReferenceBlock_j+step+1 , step ):    
                        if (reference_block_j < 0 or reference_block_j + target_block_size_j > originalFrameAxis_x) : 
                            continue   #    make sure the reference block is inside the image and not out of bounds


                    
                        #   for every reference block in the radius, check the score             

                        Score = 0
                        Count = 0

                        for i in range( 0 , target_block_size_i, step ):    #   because of the higher level, the steps will be more. We dont compare all pixels of the macroblock
                            for j in range ( 0 , target_block_size_j , step ):
                                Score += abs( int(frame2[ target_block_i + i ][ target_block_j + j ]) - int(frame1[ reference_block_i + i][ reference_block_j + j]) )
                                Count += 1
                        
                        Score = Score / Count

                        if ( Score < bestScore ):
                            bestScore = Score
                            bestScoreMacroBlock = [reference_block_i,reference_block_j]


                #   after having found the best reference block, change the motion vector
                MotionVector[macroblock_i][macroblock_j] = bestScoreMacroBlock


        
        
    


    return MotionVector













def buildPredictedFrame(PreviousFrame , MotionVector , BlockSize = 64):

    Predicted_Frame = np.zeros(PreviousFrame.shape , dtype=np.uint8)
    originalFrameAxis_x = PreviousFrame.shape[1]
    originalFrameAxis_y = PreviousFrame.shape[0]
    

    for macroblock_i in range( MotionVector.shape[0] ):

        #   we have to define block size for every macroblock because last rows/columns may have different size
        target_block_i = macroblock_i * BlockSize
        target_block_size_i = BlockSize
        if (macroblock_i == MotionVector.shape[0] - 1):
            #   we are on the last row with macroblock. the size may differ
            target_block_size_i = originalFrameAxis_y - target_block_i 

        for macroblock_j in range( MotionVector.shape[1] ):

            target_block_j = macroblock_j * BlockSize
            target_block_size_j = BlockSize
            if (macroblock_j == MotionVector.shape[1] - 1):
                #   we are on the last row with macroblock. the size may differ
                target_block_size_j = originalFrameAxis_x - target_block_j 
            

            reference_block_i = int(MotionVector[macroblock_i][macroblock_j][0])
            reference_block_j = int(MotionVector[macroblock_i][macroblock_j][1])


            PredictedBlock = PreviousFrame[ reference_block_i : reference_block_i+target_block_size_i , reference_block_j : reference_block_j+target_block_size_j ]
            



            Predicted_Frame[ target_block_i : target_block_i+target_block_size_i , target_block_j : target_block_j+target_block_size_j ] = PredictedBlock



    return Predicted_Frame            
            
            













def encode(filename):

    #   reading the video file
    Video = cv2.VideoCapture(filename)
    fps = Video.get(cv2.CAP_PROP_FPS)

    VideoFrames = []
    ret, frame = Video.read()
    while ret:
        VideoFrames.append(frame)
        ret, frame = Video.read()       



    #   producing the motion vectors and the predicted frames
    MotionVectors = []
    for frame_i in range(1,len(VideoFrames)):
        #   for each frame starting from the second, produce the motion vectors
        print(frame_i)
        MotionVectors.append(calculateMotionVector( VideoFrames[frame_i-1] , VideoFrames[frame_i] ))    
    
    PredictedFrames = []
    PredictedFrames.append(VideoFrames[0])
    for frame_i in range( 0 , len(VideoFrames)-1 ):
        #   for each frame starting from the second, produce the motion vectors
        PredictedFrames.append( buildPredictedFrame(  VideoFrames[frame_i] , MotionVectors[frame_i] ) ) 

    #   Visualizing the predicted frame sequence and saving it
    output_filename = "Predicted_Video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height = len(PredictedFrames[0])
    width = len(PredictedFrames[0][0])
    Video = cv2.VideoWriter( output_filename , fourcc , 30.0 , [width,height] )    

    for i in range( len(PredictedFrames) ):
        Video.write( PredictedFrames[i] )    
    Video.release()
    



    #   Producing the Residual Frame Using the Video Frames and the Predicted Frame
    ResidualFrames = []
    ResidualFrames.append(VideoFrames[0])
    for i in range( 1, len(PredictedFrames)):
        ResidualFrames.append( cv2.subtract(PredictedFrames[i] , VideoFrames[i]) )

    #   Visualizing the residual frame sequence and saving it
    output_filename = "Residual_Frame_Video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height = len(ResidualFrames[0])
    width = len(ResidualFrames[0][0])
    Video = cv2.VideoWriter( output_filename , fourcc , 30.0 , [width,height] )    

    for i in range( len(ResidualFrames) ):
        Video.write( ResidualFrames[i] )    
    Video.release()
    
    return [ fps , ResidualFrames , MotionVectors ]



    
    
def decode(DecodedVideo):

    fps = DecodedVideo[0]
    ResidualFrames = DecodedVideo[1]
    MotionVectors = DecodedVideo[2]

    Video = ResidualFrames
    for frame_i in range( 0 , len(ResidualFrames) - 1 ):
        PredictedFrame = buildPredictedFrame(ResidualFrames[frame_i], MotionVectors[frame_i] )
        Video[frame_i+1] = PredictedFrame + ResidualFrames[frame_i+1]
    
    #   Visualizing and saving the video
    output_filename = "Decoded_Video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height = len(Video[0])
    width = len(Video[0][0])
    Video = cv2.VideoWriter( output_filename , fourcc , fps , [width,height] )    

    for i in range( len(Video) ):
        Video.write( Video[i] )    
        cv2.imshow('video',Video[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    Video.release()

    return Video















frame1 = cv2.imread("New Folder/frame1.jpg")
frame2 = cv2.imread("New Folder/frame2.jpg")


MotionVector = calculateMotionVector( frame1, frame2)

Predictedframe = buildPredictedFrame( frame1,MotionVector )

#encodedVideo = encode("video.mp4")
#decodedVideo = decode(encodedVideo)


cv2.imshow("frame" , Predictedframe)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("test2.jpg" , Predictedframe)

