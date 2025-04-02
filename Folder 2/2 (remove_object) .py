import cv2
import numpy as np
import math




def calculateLastLevelMotionVectors(frame1, frame2, Highest_Level = 4, BlockSize = 32, Radius_k = 32):


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





def calculateMotionVector(frame1, frame2, Highest_Level = 4, BlockSize = 32, Radius_k = 32):


    

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



def buildPredictedFrame(PreviousFrame , MotionVector , BlockSize = 32):

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






def vizualizeMask( frame , mask , BlockSize = 32):
    for i in range( mask.shape[0] ):
        for j in range( mask.shape[1] ):
            if mask[i][j] == 1:
             frame[ i*BlockSize:i*BlockSize+BlockSize , j*BlockSize:j*BlockSize+BlockSize ] = 0

    return frame









def detectMovingObject(VideoFrames, FirstFrameMask, BlockSize = 32):




    #   producing the motion vectors and the predicted frames
    MotionVectors = []
        

    All_Frames_Mask = [FirstFrameMask]
    

    for frame_i in range(1,len(VideoFrames)):

        #   for each frame starting from the second, produce the motion vectors
        print(frame_i)
        MotionVectors.append(calculateMotionVector( VideoFrames[frame_i-1] , VideoFrames[frame_i] ))  

        NextFrameMask = np.zeros(FirstFrameMask.shape)
        

        for macroblock_i in range( NextFrameMask.shape[0] ):
            for macroblock_j in range( NextFrameMask.shape[1] ):

                #   finding which blocks of the previous frame are used for this macrblock
                tempblock = MotionVectors [frame_i-1] [macroblock_i] [macroblock_j]
                
                #   checking the macroblocks to which the corner belongs to

                tempBlock_i = tempblock[0] / BlockSize        
                tempBlock_j = tempblock[1] / BlockSize

                

                
                if ( tempblock[0] / BlockSize != math.floor(tempblock[0] / BlockSize) ):
                    tempBlock_i = [int(tempblock[0] / BlockSize) , math.ceil(tempblock[0] / BlockSize)]
                else:
                    tempBlock_i = [int(tempblock[0] / BlockSize)]

                
                if ( tempblock[1] / BlockSize != math.floor(tempblock[1] / BlockSize) ):
                    tempBlock_j = [ int(tempblock[1] / BlockSize) , math.ceil(tempblock[1] / BlockSize)]
                else:
                    tempBlock_j = [int(tempblock[1] / BlockSize)]


                tempBlock_Has_the_object = False
                for i in tempBlock_i:
                    for j in tempBlock_j:
                        if ( All_Frames_Mask[frame_i-1] [i][j] == 1):
                            tempBlock_Has_the_object = True
                            break
                            
                
                if (tempBlock_Has_the_object):
                    NextFrameMask[macroblock_i][macroblock_j] = 1
        
        All_Frames_Mask.append(NextFrameMask)
    
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    return All_Frames_Mask , MotionVectors



















video = cv2.VideoCapture("video1.mp4")
FrameSequence = []
ret, frame = video.read()
while ret:
    FrameSequence.append(frame)
    ret, frame = video.read()

frame1 = FrameSequence[0]



number_of_macroblocks_i = math.ceil(frame1.shape[0] / 32)
number_of_macroblocks_j = math.ceil( frame1.shape[1] / 32)
FirstFrameMask = np.zeros( [number_of_macroblocks_i,number_of_macroblocks_j] )
FirstFrameMask[ 3:10 , 7:14] = 1


#AllFramesMask = detectMovingObject(FrameSequence , FirstFrameMask)
AllFramesMask, MotionVectors = detectMovingObject(FrameSequence , FirstFrameMask)

for i in range(len(FrameSequence)):

    FrameSequence[i] = vizualizeMask(FrameSequence[i] , AllFramesMask[i])

    cv2.imshow('video',FrameSequence[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

