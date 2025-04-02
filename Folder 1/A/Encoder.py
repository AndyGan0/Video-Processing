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

        ResidualFrame = CurrentFrame.astype(int) - previousFrame.astype(int)
        videoFrames.append(ResidualFrame)
        previousFrame = CurrentFrame

    return videoFrames











def calculateHuffmanCodesDict(HuffmanTree, prefix):
    #   Gets as parameter a huffman tree and a prefix 
    #   This function calls itself recursively for every sub-list inside the Huffman tree (which is represented by a list)
    #   When calling itself, it provides the prefix, so when being called outside, no prefix should be provided


    Dict = {}

    #   Calculate nodes on the first list
    if ( len(HuffmanTree[1]) == 2): #   if first node is just a node with value then add it to dict with prefix + 1
        Dict[ HuffmanTree[1][1] ] = prefix + "1"
    else:   #   if the first node is connecting more nodes together, then call calculateHuffmanCodeDicts for that node and pass the prefix
        Dict = calculateHuffmanCodesDict(HuffmanTree[1] , prefix + "1")

    #   Calculate nodes on the second list
    if ( len(HuffmanTree[2]) == 2): #   if second node is just a node with value then add it to dict with prefix + 0
        Dict[ HuffmanTree[2][1] ] = prefix + "0"
    else:   #   if the first node is connecting more nodes together, then call calculateHuffmanCodeDicts for that node and pass the prefix
        Dict.update( calculateHuffmanCodesDict(HuffmanTree[2] , prefix + "0") )

    return Dict

    









def HuffmanEncodeFrame(Frame):
    #   Gets a givven frame and encodes it using huffman
    #   Returns the frame encoded using huffman and a dictionary to decode the frame later


    #   Creating a dictionary with the frequencies of each value
    Frequencies = {}
    for i in range(-255,256):
        Frequencies[i] = 0

    for i in range(len(Frame)):
        for j in range(len(Frame[0])):
            for k in range(len(Frame[0][0])):                
                Frequencies[Frame[i][j][k]] += 1
    

    HuffmanTree = []
    for i in range(-255,256):
        if (Frequencies[i] != 0):            
            HuffmanTree.append([Frequencies[i] , i])   
        
    HuffmanTree = sorted(HuffmanTree, key=lambda x: x[0] , reverse = True)

    #   making the huffman tree
    #   the huffman tree will be represented by nested lists instead of nodes
    #   a node(list) with a value will contain the frequency and the value
    #   after conntecting 2 nodes(lists) toegther, they will be replaced by a list which contains the sum of the frequency and the 2 nodes

    
    #   conntecting nodes(lists) in huffman tree

    while (len(HuffmanTree) != 1):      # until all nodes are conntect
        #connect the ones at the bottom with the lowest frequency

        temp1 = HuffmanTree[-2]
        temp2 = HuffmanTree[-1]     
        if (temp1[0] >= temp2[0]):  #   if temp1 has smaller frequency
            connectedNodes = [ temp1[0]+temp2[0] , temp1 , temp2 ] 
        if (temp1[0] > temp2[0]):   #   if temp1 has bigger frequency
             connectedNodes = [ temp1[0]+temp2[0] , temp2 , temp1 ] 
        HuffmanTree.pop(-1)
        HuffmanTree.pop(-1)

        #   finding the index that should be inserted based on the frequency
        index = len(HuffmanTree)
        while ( index != 0 and HuffmanTree[index-1][0] < connectedNodes[0] ):
            index -= 1
        

        HuffmanTree.insert( index , connectedNodes)

        
    HuffmanDictCodes = {}
    #   now we have the huffman tree
    #   Time to make the dictionary codes
    if( len(HuffmanTree[0]) == 2):
        #   there is only one pixel value, it will get value 1
        HuffmanDictCodes[HuffmanTree[1]] = "1"
    else:
        HuffmanDictCodes = calculateHuffmanCodesDict(HuffmanTree[0] , "")

    
    #   at this point we have the Huffman Codes Dict, we can encode the symbols
    #   the encoded frame will be a list of bool variables, where true represents 1 and false 0
    Huffman_Encoded_Frame = []
    for i in range(Frame.shape[0]):
        for j in range(Frame.shape[1]):
            for k in range(Frame.shape[2]):
                symbolForEncoding = int(Frame[i][j][k])
                EncodedPixel = HuffmanDictCodes[ symbolForEncoding ]
                for bit in EncodedPixel:
                    if (bit == '1'):
                        Huffman_Encoded_Frame.append(True)
                    else:
                        Huffman_Encoded_Frame.append(False)
    

    #   invert the dictionary so its easy to use for the decoder (swap keys and their values)
    HuffmanDictCodes = {v: k for k, v in HuffmanDictCodes.items()}


    return [Huffman_Encoded_Frame , HuffmanDictCodes]


    





    
    

    

def encode(filename):
    #   Gets the file name of a video that is in the same folder as the encoder.py file
    #   Produces the residual frames and encodes the file using huffman technique
    #   the output file is exported in the same directory
    

    # getting video information
    video = cv2.VideoCapture(filename)

    framerate = video.get(cv2.CAP_PROP_FPS)

    videoFrames = calculateResidualFrames(video)

    shape = videoFrames[0].shape

    for i in range( len(videoFrames) ):
        videoFrames[i] = HuffmanEncodeFrame(videoFrames[i])
    

    EncodedVideoFile = [videoFrames , shape , framerate]
    file1 = open("EncodedVideo.txt", "w")
    file1.write( json.dumps(EncodedVideoFile) )
    file1.close()

    #   lst = json.loads(string)

    return EncodedVideoFile


    






    

    
img = cv2.imread('1.JPG', cv2.IMREAD_COLOR) 
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (i<100):
            img[i][j] = [0,0,0]
        elif (i<300):
            img[i][j] = [30,100,50]
        else:
            img[i][j] = [240,240,200]




cv2.imshow("",img)
cv2.waitKey(0)
cv2.destroyAllWindows()



EncodedVideo = encode("video.mp4")



'''


cv2.imshow('Image', ResidualFrame)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''