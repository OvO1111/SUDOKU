from pyimagesearch.sudoku import extract_digit
import torch
import numpy as np
import cv2, os, math
import matplotlib.pyplot as plt
import pytesseract
from tensorflow.keras.preprocessing.image import img_to_array

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'


def ocr(stepX, stepY, warped, args, model, model_type, reOCR=0, pos: tuple = None):
    board = np.zeros((9, 9), dtype=int)
    # initialize a list to store the (x, y)-coordinates of each cell location
    cellLocs = []
    
    _use_second_pos = pos[reOCR] if reOCR > 0 else (0, 0)
    
    for x in range(0, 9):
        # initialize the current list of cell locations
        row = []
        
        for y in range(0, 9):
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            flag = False
            startX = y * stepX
            startY = x * stepY
            endX = (y + 1) * stepX
            endY = (x + 1) * stepY
            
            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))
            
            if (x, y) == (0, 1):
                flag = True
            # crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=args["debug"] > 0)
            
            # verify that the digit is not empty
            if digit is not None:
                # resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                if model_type == 2:  # PyTorch
                    digit = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    roi = cv2.resize(digit, (32, 32))
                    roi = roi.astype("float") / 255.0
                    roi = np.reshape(roi, (1, 1, 32, 32))
                    # expand the dimensions of roi from (28, 28) to (1, 28, 28, 1)
                    
                    # classify the digit and update the sudoku board with the prediction
                    roi = torch.from_numpy(roi)
                    model = model.double()

                    _, pred = model(roi).detach().topk(2, dim=-1, largest=True)
                    board[x, y] = pred[0].numpy()[0] if pred[0].numpy()[0] not in (0, 10) else pred[0].numpy()[1]
                    board[x, y] = board[x, y] % 10
                    
                elif model_type == 1:  # TensorFlow
                    roi = cv2.resize(digit, (28, 28))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    
                    pred = model.predict(roi).argmax(axis=1)[0]
                    board[x, y] = pred
                
                else:  # PyTesseract
                    pred = pytesseract.image_to_string(digit)
                    board[x, y] = int(pred)
        
        # add the row to our cell locations
        cellLocs.append(row)
    
    return cellLocs, board.astype(dtype=int)


def putText(board, sol, color_k=(0, 0, 0), color_u=(0, 0, 255), unfilled=None):
    """
    the function of putting numbers on a grid pic
    :param unfilled:
    :param sol: the fully filled board
    :param board: problem board
    :param color_k: color for known numbers in BGR (black)
    :param color_u: color for unknown numbers in BGR (red)
    :return: None
    """
    path = 'output/result.png'
    img = np.full((320, 320, 3), 255, dtype=float)
    shape = img.shape
    for i in range(len(board)):  # vertical lines
        ptStart = (0, shape[1] * i // 9)
        ptEnd = (shape[0], shape[1] * i // 9)
        point_color = color_k
        thickness = 1
        lineType = 4
        cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    
    for i in range(len(board)):  # horizontal lines
        ptStart = (shape[0] * i // 9, 0)
        ptEnd = (shape[0] * i // 9, shape[1])
        point_color = color_k
        thickness = 1
        lineType = 4
        cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    
    cv2.imwrite(path, img)
    
    img = cv2.imread(path)
    shape = img.shape
    if not np.any(sol):
        for x in range(len(board)):
            for y in range(len(board[0])):
                digit = board[x, y]
                textX = int(0.33 * shape[0] * (y + 1) / 3 - shape[0] / 16)
                textY = int(0.33 * shape[1] * (x + 1) / 3 - shape[0] / 64)
                if board[x, y] != 0:
                    cv2.putText(img, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, .9, color_k, 2, cv2.LINE_AA)
    else:
        for x in range(len(board)):
            for y in range(len(board[0])):
                digit = sol[x, y]
                textX = int(0.33 * shape[0] * (y + 1) / 3 - shape[0] / 16)
                textY = int(0.33 * shape[1] * (x + 1) / 3 - shape[0] / 64)
                if board[x, y] != 0:
                    cv2.putText(img, str(digit), (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX, .9, color_k, 2, cv2.LINE_AA)
                else:
                    if unfilled and x == unfilled[0] and y == unfilled[1]:
                        cv2.putText(img, str('U'), (textX, textY),
                                    cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 0, 255), 2, cv2.LINE_AA)
                        continue
                    if digit != 0:
                        cv2.putText(img, str(digit), (textX, textY),
                                    cv2.FONT_HERSHEY_SIMPLEX, .9, color_u, 2, cv2.LINE_AA)
    
    cv2.imwrite(path, img)
    if np.any(sol):
        cv2.imshow("Sudoku Result", img)
        cv2.waitKey(0)


def save_figure(print_exp, loss_mean, loss_detail):
    plt.clf()
    plt.close("all")
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('Average Loss Trend with all epochs')
    plt.plot(loss_mean)
    plt.subplot(122)
    plt.title('Detailed Loss Trend in one epoch')
    legends = []
    for i in range(print_exp):
        plt.plot(loss_detail[i])
        legends.append(f'Loss Trend in {2 ** i} epoch')
    plt.legend(legends)
    k = 0
    for _, _, e in os.walk("output/loss"):
        k = len(e)
    plt.savefig("output/loss/loss_trend_in_" + str(len(loss_mean)) + "epoch_itr" + str(k) + ".png")


def network_test(model, fig_path, debug):
    fig = cv2.imread(fig_path)
    exh = cv2.resize(fig, (280, 280))
    exh = cv2.threshold(exh, 0, 255, cv2.THRESH_BINARY)[1]
    img = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]
    roi = cv2.resize(img, (32, 32))
    roi = roi.astype("float") / 255.0
    roi = np.reshape(roi, (1, 1, 32, 32))
    roi = torch.from_numpy(roi)
    model = model.double()

    _, prd = model(roi).detach().topk(2, dim=-1, largest=True)
    pred = prd[0].numpy()[0] if prd[0].numpy()[0] not in (0, 10) else prd[0].numpy()[1]
    pred = pred % 10
    cv2.putText(exh, str(pred), (int(0.85 * exh.shape[0]), int(0.8 * exh.shape[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(fig_path + '.png', exh)
    if debug > 0:
        cv2.imshow(f"OCR Result of {fig_path}", exh)
        cv2.waitKey(0)
    return pred


def visualization(name: str, debug: int, fig_path='data/img', output_path='output/'):
    le = []
    figs = []
    for root, _, filenames in os.walk(fig_path):
        for filename in filenames:
            fig = cv2.imread(root + '/' + filename)
            if len(filename) not in le:
                le.append(len(filename))
                figs.append(fig)
                if 'png.' in filename:
                    os.remove(root + '/' + filename)
            else:
                pos = le.index(len(filename))
                try:
                    figs[pos] = np.hstack([figs[pos], fig])
                except ValueError:
                    pass
                finally:
                    if 'png.' in filename:
                        os.remove(root + '/' + filename)
    
    for i in range(2, len(figs)):
        try:
            figs[1] = np.vstack([figs[1], figs[i]])
        except (IndexError, ValueError):
            continue
    
    cv2.imwrite(output_path + name + '.png', figs[1])
    if debug > 0:
        cv2.imshow(f'cpr.png', figs[1])
        cv2.waitKey(0)


def convert(a: list):
    return [i for i in a]


def generator(pos: float, itr: int):
    return [pos for i in range(itr)]
