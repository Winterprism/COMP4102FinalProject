# Comp4102 Final project Distance calculator
# Kevin Zhang 101148146
# Ning Hu 101151560
# Satsang Adhikari 101145635


# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

fontSize = 0.55


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.equalizeHist(gray)
blur = cv2.GaussianBlur(hist, (31,31), cv2.BORDER_DEFAULT)
height, width = blur.shape[:2]

minR = round(width/65)
maxR = round(width/11)
minDis = round(width/7)


circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDis, param1=100, param2=25, minRadius=minR, maxRadius=maxR)


# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)


# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# compute the rotated bounding box of the contour
orig = image.copy()


# stores the centerpoint for each image
objectCenterPoints = []


# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue

	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	
	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	(tX, tY) = midpoint(tl, br)
	cv2.circle(orig, (int(tX), int(tY)), 5, (0, 0, 0), -1)

	circles = np.asarray(circles)

	if circles.ndim == 2:
		circles = [circles]

	
	circleFound = False
	# Detects if a circle is within the image
	if circles is not None:
		for ci in circles[0]:
			# Does not draw a rectange if the center point is less than or equal to 6 pixels aways from center at circle ci
			if abs(ci[0] - tX) <= 6 and abs(ci[1] - tY) <= 6:
				circleFound = True
				circles = np.round(circles[0, :]).astype("int")
				
				cA = dist.euclidean((ci[0] - ci[2],ci[1]), (ci[0] + ci[2],ci[1]))

				# if the pixels per metric has not been initialized, then
				# compute it as the ratio of pixels to supplied metric
				# (in this case, inches)
				if pixelsPerMetric is None:
					pixelsPerMetric = cA / args["width"]

				# compute the size of the object
				dimcA = cA / pixelsPerMetric

				# Displays diameter over circles found
				cv2.putText(orig, "{:.1f}in".format(dimcA), (int(ci[0]-25),int(ci[1]-ci[2]-10)),cv2.FONT_HERSHEY_SIMPLEX,fontSize, (200, 0, 200), 2)

				for (x, y, r) in circles:
					cv2.rectangle(orig, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 0), -1)
					cv2.circle(orig, (x, y), r, (0, 255, 0), 2)

				break

		if circleFound == False:
			# loop over the original points and draw them
			for (x, y) in box:
				cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
				cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
		
			cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
			cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
			cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
			cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

			# draw lines between the midpoints
			cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
				(255, 0, 255), 2)
			cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
				(255, 0, 255), 2)


						

			# compute the Euclidean distance between the midpoints
			dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
			dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

			# if the pixels per metric has not been initialized, then
			# compute it as the ratio of pixels to supplied metric
			# (in this case, inches)
			if pixelsPerMetric is None:
				pixelsPerMetric = dB / args["width"]

			# compute the size of the object
			dimA = dA / pixelsPerMetric
			dimB = dB / pixelsPerMetric

			# draw the object sizes on the image
			cv2.putText(orig, "{:.1f}in".format(dimA),
				(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 250, 56), 2)
			cv2.putText(orig, "{:.1f}in".format(dimB),
				(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 250, 56), 2)
			
	objectCenterPoints.append((int(tX), int(tY)))



for x in range(len(objectCenterPoints)-1):
	dm = dist.euclidean((objectCenterPoints[x][0]   ,objectCenterPoints[x][1]), (objectCenterPoints[x+1][0],objectCenterPoints[x+1][1]))
	dmA = dm / pixelsPerMetric
	
	cv2.line(orig,objectCenterPoints[x], objectCenterPoints[x+1],(85,200,255), 2)
	# (85,200,255)
	# (35,100,125)
	# (30,30,50) Winner
 
	# Displays the distance between objects
	cv2.putText(orig, "{:.1f}in".format(dmA ), (int((objectCenterPoints[x][0] + objectCenterPoints[x+1][0])/2),int((objectCenterPoints[x][1]+objectCenterPoints[x+1][1])/2)),cv2.FONT_HERSHEY_SIMPLEX,fontSize, (0, 0, 255), 2)

	
# # show the output image
cv2.imshow("Image", orig)
cv2.waitKey(0)

