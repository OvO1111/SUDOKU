# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


def find_puzzle(image, debug=False):
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    
    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    
    # check to see if we are visualizing each step of the image
    # processing pipeline (in this case, thresholding)
    if debug > 0:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)
    
    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_EXTERNAL=only search for outer-most contour
    # cv2.CHAIN_APPROX_SIMPLE=zip the contour info to only the vertices
    # (e.g. a rectangle contour=4 vertices, a triangle contour=3 v.)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None
    
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)  # the contour circumstance, True indicates a closed contour
        approx = cv2.approxPolyDP(c, epsilon=0.02 * peri, closed=True)
        # if using new edge E directly connect the starting and ending point
        # (in this case, they are the same because of a closed contour),
        # results in a point P whose distance with respect to E is
        # larger than epsilon, then establish two edges E1, E2 by connecting
        # starting point and ending point with P. Recurse this operation until
        # the epsilon threshold is satisfied
        
        # if our approximated contour has four points (a rect in reduced form), then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break
    
    # if the puzzle contour is empty then our script could not find
    # the outline of the sudoku puzzle so raise an error
    if puzzleCnt is None:
        raise Exception(("Could not find sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))
    
    # check to see if we are visualizing the outline of the detected
    # sudoku puzzle
    if debug > 0:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)
    
    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down birds eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    # the np.reshape function is used because the original contour array is of
    # dim=(n, 1, 2): n for n vertices, 2 for 2-d picture, looks like
    # [[[1, 2]], [[0, 9]], [[1, 8]]],
    # after the approxPolyDP transformation it became (4, 1, 2)
    
    # check to see if we are visualizing the perspective transform
    if debug > 0:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    
    # return a 2-tuple of puzzle in both RGB and grayscale
    return puzzle, warped


def extract_digit(cell, debug=False):
    # val < 120 is valid
    thresh = cv2.threshold(cell, 120, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.dilate(thresh, np.ones((2, 2), dtype=np.uint8), iterations=2)
    (h, w) = thresh.shape
    
    w_l_bound = int(0.15*w)
    w_r_bound = int(0.05*w)
    h_u_bound = int(0.2*h)
    h_d_bound = int(0.1*h)
    
    thresh[0: h_u_bound] = 0
    thresh[-h_d_bound: -1] = 0
    thresh[:, 0: w_l_bound] = 0
    thresh[:, -w_r_bound: -1] = 0
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        return None
    
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    mask = np.zeros(thresh.shape, dtype="uint8")
    sm, sc = 0, 0
    for cnt in cnts:
        if cv2.contourArea(cnt) >= 1e-4*float(w * h):
            sc += cv2.contourArea(cnt)
            sm += 1
    
    for i in range(sm):
        cv2.drawContours(mask, [cnts[i]], -1, 255, -1)
    
    percentFilled = sc / (w*h)
    if percentFilled < 1e-3:
        return None
    
    # apply the mask to the threshold-ed cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    if np.all(digit == 0):
        return None
    return digit
