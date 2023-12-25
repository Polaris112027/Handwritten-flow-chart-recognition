import cv2
import numpy as np
from matplotlib import pyplot as plt

class complex_shape_recognition:
    def __init__(self, path_img, print = False):
        self.path_img = path_img
        self.image = cv2.imread(path_img)
        self.recognized_shapes = {}
        if print == True:
            plt.imshow(self.image)
            plt.show()

    def img_size(self):
        height_original, width_original, channels_original = self.image.shape
        return [height_original, width_original]
    
    def resize_img(self):
        # resize to height 600 pixels
        height_original, width_original, channels_original = self.image.shape
        height_new = 600
        width_new = int(width_original * ( height_new / height_original))
        self.scaling_ratio = ( height_new / height_original)
        shape_points = (width_new, height_new)
        self.image_resized = cv2.resize(self.image, shape_points, interpolation= cv2.INTER_LINEAR)

    def gray_img(self):
        # convert to gray
        self.gray = cv2.cvtColor(self.image_resized, cv2.COLOR_BGR2GRAY)

    def inverse_img(self):
        # inverse
        self.Inverse_gray = self.gray.copy()
        height, width = self.Inverse_gray.shape
        for i in range(height):
            for j in range(width):
                pv = self.Inverse_gray[i, j]
                self.Inverse_gray[i][j] = 255 - pv

    def binarize_img(self):
        # make it binary
        ret, binary = cv2.threshold(self.Inverse_gray, 75, 255, cv2.THRESH_BINARY)
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))    #make dilation and erodation if neccessary
        #dilation = cv2.dilate(binary, kernel, iterations = 3)
        #self.image = cv2.erode(dilation, kernel, iterations = 3)
        self.image = binary
        
    def img_preprocessing(self, print = False):
        self.resize_img()
        self.gray_img()
        self.inverse_img()
        self.binarize_img()
        if print == True:
            plt.imshow(self.image)
            plt.show()

    def flood_img(self, print = False):
        self.img_floodfill = self.image.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = self.image.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8) 
        # Floodfill from point (0, 0)
        cv2.floodFill(self.img_floodfill, mask, (0,0), 255)
        # Invert floodfilled image
        self.img_floodfill_inv = cv2.bitwise_not(self.img_floodfill)
        # Combine the two images to get the foreground.
        img_foreground = self.image | self.img_floodfill_inv
        # expand the shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilation = cv2.dilate(self.img_floodfill_inv, kernel, iterations = 3)
        # cut out the rest of img
        self.rest_img = img_foreground - dilation
        
        if print == True:
            plt.imshow(self.img_floodfill_inv)
            plt.show()

    def shape_detection(self, accuracy = 0.05, print = False):
        # find the contours
        contours, _ = cv2.findContours(self.img_floodfill_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(self.image_resized, contours, -1, (0, 0, 255), 2)
        image_copy = self.image_resized.copy()

        self.contour_num = 0
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 100:
                # edit the coefficient before "perimeter" to prune the accuracy of contour drawing 
                approx = cv2.approxPolyDP(contour, accuracy * perimeter, True)
                sides = len(approx)
                if sides == 3:
                    shape = "triangle"
                elif sides == 4:
                    shape = "rectangle"
                elif sides > 4:
                    shape = "circle"

                cv2.drawContours(image_copy, [approx], -1, (0, 255, 0), 2)
                cv2.putText(image_copy, shape, (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # compute the center of the contour
                M = cv2.moments(contour)
                #print(M)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.circle(image_copy, (cX, cY), 7, (0, 0, 0), -1)
                cv2.putText(image_copy, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # compute the coordinates of bounding rectangle
                x,y,w,h = cv2.boundingRect(contour)
                # scaling back to original size
                x,y,w,h = x/self.scaling_ratio, y/self.scaling_ratio, w/self.scaling_ratio, h/self.scaling_ratio
                # record into dictionary
                self.recognized_shapes["shape"+str(self.contour_num)] = [shape, (x, y, w, h)]
                self.contour_num = self.contour_num + 1
        if print == True:
            plt.imshow(image_copy)
            plt.show()

    def shape_detection_rest_img(self, print = False):     
        image_copy = self.image_resized.copy()
        #image_copy = self.rest_img.copy()
        # find the contours for the rest of img
        edges = cv2.Canny(self.rest_img.copy(), 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 100:
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                shape = "arrow"

                cv2.drawContours(image_copy, [approx], -1, (0, 255, 0), 2)
                cv2.putText(image_copy, shape, (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                x,y,w,h = cv2.boundingRect(contour)
                # scaling back to original size
                x,y,w,h = x/self.scaling_ratio, y/self.scaling_ratio, w/self.scaling_ratio, h/self.scaling_ratio
                self.recognized_shapes["shape"+str(self.contour_num)] = [shape, (x, y, w, h)]
                self.contour_num = self.contour_num + 1
            
        if print == True:
            plt.imshow(image_copy)
            plt.show()

    def shapes(self):
        return self.recognized_shapes

def shape_recognition(IMAGE_PATH):
    recognition = complex_shape_recognition(IMAGE_PATH)
    original_size = recognition.img_size()
    recognition.img_preprocessing()
    recognition.flood_img()
    # Accuracy parameter better between 0.01 and 0.05
    recognition.shape_detection(0.04)
    recognition.shape_detection_rest_img()
    # might got some duplicated result, can be reduced by checking the bounding boxes
    
    return recognition.shapes(), original_size

def toxml(dict, xml_path):
    # make list to store html content
    content_list_xml = []
    content_list_xml.append('<mxGraphModel><root><mxCell id="0" /><mxCell id="1" parent="0" />')
    # make x,y,w,h into html tag format
    for i in range(len(dict)):
        if dict["shape"+str(i)][0] == 'rectangle':
            content_list_xml.append('<mxCell id="'+'shape'+str(i)+'" value="" style="whiteSpace=wrap;html=1;aspect=fixed;fillColor=none;" vertex="1" parent="1"><mxGeometry x="'+str(int(dict["shape"+str(i)][1][0]))+'" y="'+str(int(dict["shape"+str(i)][1][1]))+'" width="'+str(int(dict["shape"+str(i)][1][2]))+'" height="'+str(int(dict["shape"+str(i)][1][3]))+'" as="geometry" /></mxCell>')
        elif dict["shape"+str(i)][0] == 'circle':
            content_list_xml.append('<mxCell id="'+'shape'+str(i)+'" value="" style="ellipse;whiteSpace=wrap;html=1;fillColor=none;" vertex="1" parent="1"><mxGeometry x="'+str(int(dict["shape"+str(i)][1][0]))+'" y="'+str(int(dict["shape"+str(i)][1][1]))+'" width="'+str(int(dict["shape"+str(i)][1][2]))+'" height="'+str(int(dict["shape"+str(i)][1][3]))+'" as="geometry" /></mxCell>')
        elif dict["shape"+str(i)][0] == 'triangle':
            content_list_xml.append('<mxCell id="'+'shape'+str(i)+'" value="" style="triangle;whiteSpace=wrap;html=1;rotation=-90;fillColor=none;" vertex="1" parent="1"><mxGeometry x="'+str(int(dict["shape"+str(i)][1][0]))+'" y="'+str(int(dict["shape"+str(i)][1][1]))+'" width="'+str(int(dict["shape"+str(i)][1][2]))+'" height="'+str(int(dict["shape"+str(i)][1][3]))+'" as="geometry" /></mxCell>')
        else:
            pass
    content_list_xml.append('</root></mxGraphModel>')
    #write html file
    with open(xml_path, 'w') as file:
        file.writelines(content_list_xml)

def shape_digitalization(IMAGE_PATH):
    xml_path = IMAGE_PATH.split(".")[0]+"_shapes.xml"
    dict, original_size = shape_recognition(IMAGE_PATH)
    toxml(dict, xml_path)