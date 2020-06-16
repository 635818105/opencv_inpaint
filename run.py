import os

import numpy as np
import cv2
from easygui import *

BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG
THICKNESS = -1  # solid brush circle 实心圆, 鼠标线条的粗细
DRAW_MASK = {'color': BLACK, 'val': 255}
drawing = False
radius = 3  # brush radius
value = DRAW_MASK
dst = None


def onmouse_input(event, x, y, flags, param):
    """
    mouse callback function, whenever mouse move or click in input window this function is called.
    只要鼠标在input窗口上移动(点击），此函数就会被回调执行
    """
    # to change the variable outside of the function
    # 为方法体外的变量赋值，声明global
    global img, img2, drawing, value, mask, dst
    # print(x,y)

    # draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img, (x, y), radius, value['color'], THICKNESS, lineType=cv2.LINE_AA)
        cv2.circle(mask, (x, y), radius, value['val'], THICKNESS, lineType=cv2.LINE_AA)

    elif drawing is True and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(img, (x, y), radius, value['color'], THICKNESS, lineType=cv2.LINE_AA)
        cv2.circle(mask, (x, y), radius, value['val'], THICKNESS, lineType=cv2.LINE_AA)

    elif drawing is True and event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), radius, value['color'], THICKNESS, lineType=cv2.LINE_AA)
        cv2.circle(mask, (x, y), radius, value['val'], THICKNESS, lineType=cv2.LINE_AA)
        dst = inpain(img, mask)
        img = dst.copy()


def inpain(input_image, mask_image):
    dst = cv2.inpaint(input_image, mask_image, 1, cv2.INPAINT_TELEA)
    # cv2.imshow('INPAINT_TELEA', dst)
    return dst


if __name__ == "__main__":
    # image: absolute image path
    image = fileopenbox(msg='Select a image', title='Select', filetypes=[['*.png', '*.jpg', '*.jpeg', 'Image Files']])
    if image is None:
        print('\nPlease select a image.')
        exit()
    else:
        print('Image selected: ' + image)
    img = cv2.imread(image)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    # resize the photo if it is too small
    if max(img.shape[0], img.shape[1]) < 256:
        img = cv2.resize(img, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img2 = img.copy()  # a copy of original image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG

    # input and output windows
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', onmouse_input)
    cv2.moveWindow('input', img.shape[1] + 20, 100)  # move input window

    while 1:
        cv2.imshow('input', img)
        cv2.imshow("mask", mask)
        k = cv2.waitKey(200)
    
        # key bindings
        if k == 27 or k == ord('q'):  # esc to exit
            break
        elif k == ord('0'):  # BG drawing
            print(" mark background regions with left mouse button \n")
            value = DRAW_MASK
        elif k == ord('r'):  # reset everything
            print("resetting \n")
            drawing = False
            value = DRAW_MASK
            img = dst.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
        elif k == ord('['):
            radius = 1 if radius == 1 else radius - 1
            print('Brush thickness is', radius)
        elif k == ord(']'):
            radius += 1
            print('Brush thickness is', radius)
        elif k == ord('s'):
            path = filesavebox('save', 'save the output.', default='patched_' + os.path.basename(image),
                               filetypes=[['*.jpg', 'jpg'], ['*.png', 'png']])
            if path:
                if not path.endswith('.jpg') and not path.endswith('.png'):
                    path = str(path) + '.png'
                cv2.imwrite(path, dst)
                print('Patched image is saved to', path)
    cv2.destroyAllWindows()


