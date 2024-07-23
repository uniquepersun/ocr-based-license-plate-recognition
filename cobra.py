import cv2
import numpy as np
import pytesseract

def preprocess_image(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  return thresh

def detect_number_plate(img):
  contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
      x, y, w, h = cv2.boundingRect(approx)
      plate_img = img[y:y+h, x:x+w]
      return plate_img
  return None

def extract_characters(plate_img):

  gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  characters = []
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    char_img = thresh[y:y+h, x:x+w]
    characters.append(char_img)
  return characters

def recognize_characters(characters):
  text = ""
  for char_img in characters:
    char_img = cv2.resize(char_img, (32, 32))
    char_img = np.pad(char_img, ((10, 10), (10, 10)), mode='constant', constant_values=255)
    try:
      text += pytesseract.image_to_string(char_img, config='--psm 10')
    except:
      text += '?'
  return text

def main():
  img = cv2.imread('car_image.jpg')
  processed_img = preprocess_image(img)
  plate_img = detect_number_plate(processed_img)
  if plate_img is not None:
    characters = extract_characters(plate_img)
    number_plate = recognize_characters(characters)
    print(number_plate)
  else:
    print("Number plate not found.")

if __name__ == "__main__":
  main()
