import cv2
import numpy as np
import os
class ImageProcessing:
    def __init__(self, image_path):
        self.image_path = image_path
        self.ocr_img = None
        self.h_src = None
        self.w_src = None
        self.c_src = None
        self.name = None

    def load_image(self):
        self.ocr_img = cv2.imread(self.image_path)
        self.h_src, self.w_src, self.c_src = self.ocr_img.shape
        self.name = os.path.splitext(os.path.basename(self.image_path))[0]
        #print('the original size:', self.h_src, self.w_src)
        #print('the image name:', os.path.basename(self.image_path))
#1. Scale the input images to a certain size, and then binarize them to adjust the grayscale.
    def resize_image(self, target_long_side):
        # 获取原始图片的宽度和高度
        height, width = self.ocr_img.shape[:2]
        if height > width:
            scale = target_long_side / height
            new_width = int(width * scale)
            new_height = target_long_side
        else:
            scale = target_long_side / width
            new_width = target_long_side
            new_height = int(height * scale)

        self.ocr_img = cv2.resize(self.ocr_img, (new_width, new_height))
        self.h_src, self.w_src = new_height, new_width
        #print('the new size:',self.h_src, self.w_src)

    def preprocess_image(self):
        ocr_img_gray = cv2.cvtColor(self.ocr_img, cv2.COLOR_BGR2GRAY)
        ocr_img_gray = cv2.GaussianBlur(ocr_img_gray, (3, 3), 1)
        ret, ocr_img_thresh = cv2.threshold(ocr_img_gray, 200, 255, cv2.THRESH_BINARY)
        self.ocr_img_thresh = ocr_img_thresh


#2. Find their outer outlines and draw them with green lines

    def find_contours(self):
        ocr_img_contours, hierarchy = cv2.findContours(self.ocr_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.ocr_img_contours = ocr_img_contours

    def process_contours(self):
        draw_img = self.ocr_img.copy()
        cont_max = sorted(self.ocr_img_contours, key=cv2.contourArea, reverse=True)[0]
        x, y, w, h = cv2.boundingRect(cont_max)
        arcLength = cv2.arcLength(cont_max, True)
        rate = 0.01
        approx_max = None
        while len(cont_max) != 4:
            approx_max = cv2.approxPolyDP(cont_max, epsilon=rate * arcLength, closed=True)
            if len(approx_max) == 4:
                break
            rate += 0.01
        #print("approx:", approx_max)
        self.draw_img = cv2.drawContours(draw_img, [approx_max], -1, color=(0, 255, 0), thickness=2)
        self.approx_max = approx_max

    def sort_dot_cnt(self, kps):
        rect = np.zeros((4, 2), dtype='float32')
        s = kps.sum(axis=1)
        rect[0] = kps[np.argmin(s)]
        rect[2] = kps[np.argmax(s)]
        diff = np.diff(kps, axis=1)
        rect[1] = kps[np.argmin(diff)]
        rect[3] = kps[np.argmax(diff)]
        return rect
#3. Carry out perspective correction

    def perspective_transform(self):
        rect_ordered = self.sort_dot_cnt(self.approx_max.reshape(4, 2))
        (top_left, top_right, bottom_right, bottom_left) = rect_ordered
        pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
        pts_dst = np.array([(0 + top_left[0], 0 + top_left[1]),
                            (self.w_src + top_left[0], 0 + top_left[1]),
                            (self.w_src + top_left[0], self.h_src + top_left[1]),
                            (0 + top_left[0], self.h_src + top_left[1])], dtype="float32")
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        im_out = cv2.warpPerspective(self.ocr_img_thresh, M, (self.w_src, self.h_src))
        self.im_out = im_out
#4. Place the corrected picture on a canvas of a certain size
    def create_canvas(self):
        canvas = np.zeros((1800, 1500), dtype=np.uint8) + 255
        h_canvas, w_canvas = canvas.shape
        h_output, w_output = self.im_out.shape
        x_offset = int((w_canvas - w_output) / 2)
        y_offset = int((h_canvas - h_output) / 2)
        canvas[y_offset:y_offset + h_output, x_offset:x_offset + w_output] = self.im_out
        self.canvas = canvas

    def save_results(self,canvas_path):
        cv2.imwrite(canvas_path, self.canvas)


    def process_image(self, target_long_side=1400, canvas_path=None):
        self.load_image()
        self.resize_image(target_long_side)
        self.preprocess_image()
        self.find_contours()
        self.process_contours()
        self.perspective_transform()
        self.create_canvas()
        if canvas_path is None:
            canvas_path = self.name + '_canvas.jpg'
        self.save_results(canvas_path)
        return canvas_path
