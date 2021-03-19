import cv2
import numpy as np
import imutils
import os

#---------------------------------------------------------------------------#
#                   Initialization of template images
#---------------------------------------------------------------------------#
ace = cv2.imread('a.png',0)
two = cv2.imread('2.png',0)
three = cv2.imread('3.png',0) 
four = cv2.imshow('4.png',0)
five =cv2.imread('5.png',0)
six = cv2.imread('6.png',0) 
seven = cv2.imread('7.png',0)
eight = cv2.imread('8.png',0)
nine = cv2.imread('9.png',0)
ten = cv2.imread('10.png',0)
j = cv2.imread('j.png',0)
q = cv2.imread('q.png',0) 
k = cv2.imread('k.png',0)
template = [ace, two, three, four, five, six, seven, eight, nine, ten, j, q, k]

size_w, size_h = four.shape

#--------------------------------  End  ------------------------------------#


#---------------------------------------------------------------------------#
#              Initialization for Rank Counting Operations
#---------------------------------------------------------------------------#
player = [] # list to record all player's total card rank information
playerNumber = 0
highest = 0
lowest = 100

# to calculate total rank and determine total covered card
def calculateRank(playerRanks):
    
    total = 0
    covered = 0
    player = []

    # loop through cards of each player
    for i in range(1,len(playerRanks)):

        # if card is None, it is a covered card
        if(playerRanks[i] == None):
            covered += 1 
        # calculate total rank
        else:
            total += playerRanks[i]
    
    # if theres no covered cards, set value as None     
    if (covered == 0):
        covered = None

    # append player type, total and total covered card to a list
    player.append(playerRanks[0])
    player.append(total)
    player.append(covered)

    return player


# to print results
def printResults(player,highest, lowest):

    flag = False
    for i in range(0,len(player)):  
        if(player[i][0] != "Dealer"):
            print ("Player" , player[i][0] , "=" , player[i][1])
        else:
            print (player[i][0] , "=" , player[i][1])

    for i in range(0, len(player)):
        # if player has highest, less than or equal 21 and have no covered cards
        # player is the winner
        if(player[i][1] == highest and player[i][1] <= 21 and player[i][2] == None):
            print ("")
            if player[i][0] == "Dealer":
                print ("Winner = " , player[i][0])
            else:
                print ("Winner = Player" , player[i][0])
            flag = True
            break
        
        # if all player has covered card, player with lowest total rank
        # that is less than 21 will have highest chance of winning
        else:
            if flag == True:
                break
           
            if (player[i][1] == lowest and player[i][1] <= 21):
                print ("")
                if player[i][0] == "Dealer":
                    print ("Winner = " , player[i][0])
                else:
                    print ("Winner = Player" , player[i][0])

        
#--------------------------------  End  ------------------------------------#



#---------------------------------------------------------------------------#
#                   Initialization and Pre-processing of images
#---------------------------------------------------------------------------#
# structuring element for sharpening image
k = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# initialize source image
image = cv2.imread('playing_cards_on_table.bmp')

# rotate source image to 180 degrees
image = cv2.flip( image, -1 )

# get dimensions of source image
width, height, _ = image.shape

# resize image 
resize = cv2.resize(image, (3900, 1550))

# preprocess image
gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
retval, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

# Perform morphological operations based on image dimensions
if width > 3900:
    # morphologically close all small details in the cards to avoid false contours later
    # then dilate the image to segment out individual players
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    morphed = cv2.dilate(morphed, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations = 8)
else:
    # morphologically close all small details in the cards to avoid false contours later
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None, iterations=12)

# sharpen the image if it is smaller image
if width < 3900:
    resize = cv2.filter2D(resize, -1, k)

# find contours, use CCOMP for smaller image and EXTERNAL for larger image
if width > 3900:
    _,cnts,hier = cv2.findContours(morphed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
else:
    _,cnts,hier = cv2.findContours(morphed,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

# sum up all contour areas to find average area size
areasum = 0

for i in range(len(cnts)):
    areasum = areasum + cv2.contourArea(cnts[i])
    
avgarea = areasum/len(cnts)

#-----------------------------------  End  --------------------------------------#


#--------------------------------------------------------------------------------#
#                               Main Loop
#--------------------------------------------------------------------------------#

# loop through all contours ignore contours larger than average contour area
for i in range(len(cnts)):
    
    # contours that are smaller than average area are individual players
    if cv2.contourArea(cnts[i]) < avgarea and cv2.contourArea(cnts[i]) > 30000:
        
        #list for each player's card rank information
        playerRank = []
        
        # obtain dimensions of contour
        x,y,w,h = cv2.boundingRect(cnts[i])
        
        # crop out contour segment, threshold them
        seg = resize[y:y+h,x:x+w]       
        g = cv2.cvtColor(seg,cv2.COLOR_BGR2GRAY)
        _,segment = cv2.threshold(g,100,255,cv2.THRESH_BINARY)
        
        # erode the contour for more distinct segmentation
        segment = cv2.erode(segment, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)), iterations = 1)
        segment = cv2.morphologyEx(segment, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)), iterations = 1)
        
        # get dimensions of the contour segment
        col,row = segment.shape[::-1]         
            
        # find individual card contours in segment
        _,contours,h = cv2.findContours(segment,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)
        
        # dealer hand contour has larger width
        if(col > 800):
            playerRank.append("Dealer")
        else:
            playerNumber +=1
            playerRank.append(playerNumber)
        
        # Loop through each detected card contours in segment
        for i in range(len(contours)):
            
            # ignore small contours as they are not cards
            if cv2.contourArea(contours[i]) > 5000:               
                
                # get the minimum enclosing rectangle of each card contour
                (x2,y2),(w2,h2),angle = cv2.minAreaRect(contours[i])
                rect = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                p1,p2,p3,p4 = box
                
                x2 = int(x2)
                y2 = int(y2)
                h2 = int(h2)
                w2 = int(w2)
           
                # based on the angle obtained from minimum enclosing rectangle, 
                # calculate rotation matrix of each card contour.
                # Warp each card contour to its upright angle.
                rotateMatrix = cv2.getRotationMatrix2D((x2,y2),angle,1)
                rotated = cv2.warpAffine(seg,rotateMatrix,(col,row))
                rotated = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
                
                # rotate the vertice points of each card contour to its 
                # upright angle and get the new coordinates of each rotated point
                pts = np.int0(cv2.transform(np.array([box]), rotateMatrix))[0]
                pts[pts < 0] = 0
                
                # crop out each card segment based on the rotated points
                crop = rotated[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]
                
                # obtain cropped image dimensions
                cropx,cropy = crop.shape[::-1]
                
                # if crop is oriented vertically, rotate 270 degrees to make it vertical
                if cropx > cropy:
                    crop = imutils.rotate_bound(crop, 270)
                
                cv2.imshow("res", crop)
                cv2.waitKey()
                cv2.destroyAllWindows()

                count = 0         
                flag = False
                checker = 0

                # Perform template matching between card and template
                # Loop through the 13 template images
                for temp in range(len(template)):
                    
                    # to keep track of templates gone through
                    checker += 1
                    
                    res = cv2.matchTemplate(crop,template[temp],cv2.TM_CCOEFF_NORMED)        
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    
                    # get matched card rank value
                    if temp < 10:
                        count += 1
                    else:
                        count = 10
                        
                    # only execute if matching rate is equal to or more than 80%
                    if max_val>=0.8:
                        
                        flag = True 
                        playerRank.append(count) # insert rank value into list 
                    
                    # if all templates have been gone through and still no match
                    # the contour is considered a covered card
                    if checker == 13 and flag == False:
                        playerRank.append(None)
        
        # the list with each player's card rank is parsed to count rank and determine total covered cards
        res = calculateRank(playerRank)
        player.append(res) #results are appended to a list


# loop through all players to find hand with highest total rank
for i in range(0,len(player)):
    # if total is less than or equals to 21 and has no covered cards, its the highest
    if(player[i][1] >= highest and player[i][1] <= 21 and player[i][2] == None ):
        highest = player[i][1]
        
# loop through all players to find hand with lowest total rank
for i in range(0,len(player)):
    # lowest total rank that is not 0 will be chosen as lowest value
    if(player[i][1] <= lowest and player[i][1] <= 21 and player[i][1] != 0):
        lowest = player[i][1]

# Print results
printResults(player, highest, lowest)

#---------------------------------  End  -----------------------------------------#

