# USAGE
# python solve_sudoku_puzzle.py --model output/digit_classifier.h5 --image img/1.png

# import the necessary packages
from pyimagesearch.sudoku import find_puzzle
import torch
from sudoku import Sudoku
import utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2, os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained digit classifier")
ap.add_argument("-i", "--image", required=True,
                help="path to input sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

# simplify args:
file, itr = 'torch0.ckpt', 0
for _, _, filenames in os.walk('output/'):
    for filename in filenames:
        if '.ckpt' in filename:
            if int(filename.lstrip('torch').rstrip('.ckpt')) > itr:
                file = filename
                itr = int(filename.lstrip('torch').rstrip('.ckpt'))

file = 'output/'+file

args = {"py-model": file, "t-model": "output/digit_classifier.h5", "image": "img/" + '2-3.jpg', "debug": 0}
# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = torch.load(args["py-model"])

'''print("[INFO] loading digit classifier...")
model = torch.load(args["model"])'''

model_type = 2

# load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

# find the puzzle in the image and then
(puzzleImage, warped) = find_puzzle(image, debug=args["debug"])

stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9


def main():
    # loop over the grid locations
    cellLocs, board = \
        utils.ocr(stepX, stepY, warped, args, model, model_type)
    
    # construct a sudoku puzzle from the board
    utils.putText(board, sol=None)
    im1 = cv2.imread('output/result.png')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    mid = np.ones((320, 10))
    im2 = np.hstack([im1, mid, cv2.resize(warped, (320, 320))])
    cv2.imwrite('output/result.png', im2)
    puzzle = Sudoku(gridMap=board)
    puzzle.show_full()
    
    # solve the sudoku puzzle
    solution = puzzle.solution
    print("[INFO] solving sudoku puzzle...")
    puzzle.show_full()
    
    utils.putText(board, solution, unfilled=puzzle.unfillable)


if __name__ == '__main__':
    main()
