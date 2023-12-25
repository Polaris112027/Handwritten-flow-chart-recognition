import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw
import numpy as np
import cv2
import json
# global variables
reader = None
processor = None
model = None

def initialize_ocr():
    global reader, processor, model
    # Check if global variables have been initialized
    if reader is None:
        # Initialize OCR reader and model
        reader = easyocr.Reader(['ru','en'])
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def recognize(image_path, corner_radius=0.6):
    # Initialize OCR reader and model
    initialize_ocr()
    
    # Make filename
    filename, ext = image_path.split('.')
    coordinates_text_path = filename + '_coordinates_text' +'.txt'
    cut_image_path = filename + '_cut.' + ext
    xml_path=filename + '_text.xml'
    # Read text from image
    result = reader.readtext(image_path)
    coords_list = [x[0] for x in result]
    

    # Replace text regions with white rectangles
    image = Image.open(image_path).convert("RGB")
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    for coords in coords_list:
        x1, y1 = coords[0]
        x2, y2 = coords[2]
        draw.rounded_rectangle([(x1, y1), (x2, y2)], corner_radius*100, fill=(255, 255, 255, 255))

    # Save the modified image
    new_image.save(cut_image_path)
    print("Completed cutting out handwritten text from image")

    # Extract text from modified image
    recognized_text = []
    for coords in coords_list:
        x1, y1 = coords[0]
        x2, y2 = coords[2]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        coords = (x1, y1, x2, y2)
        cropped_image = image.crop(coords)
        pixel_values = processor(cropped_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        recognized_text.append({"coordinates": coords, "text": generated_text})
        text_str = str(recognized_text)
        with open(coordinates_text_path, 'w') as f:
            f.write(text_str)
    print("Completed handwritten text recognition in image")

    width, height = image.size
    # Read and parse the recognition result file
    with open(coordinates_text_path, "r") as f:
        text_str = f.read()
        text_str = text_str.replace("'", '"')
        text_str = text_str.replace("(", "[").replace(")", "]")
        text = json.loads(text_str)

    # Traverse text data and generate corresponding SVG tags based on coordinates and text
    text_tags = f"""<svg version="1.1" baseProfile="full" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width={width} height={height} >"""
    for item in text:
        x1, y1, x2, y2 = item['coordinates']
        text = item['text']
        text_tag = f"<text x='{x1}' y='{y1}'font-size='30'>{text}</text>"
        text_tags += text_tag
    # Save final XML code to file
    with open(xml_path, 'w') as file:
        file.write(text_tags + "</svg>")
    print("Complete text xml file")
#recognize_and_cut(test5.jpg)