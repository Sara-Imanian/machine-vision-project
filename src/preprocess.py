import cv2
import numpy as np 
from pathlib import Path

IMG_SIZE = (256,256)

def load_and_preprocess(path):
    img = cv2.imread(str(path))
    if img is None :
        raise FileNotFoundError(f"Image not found at {path}")

    img = cv2.resize(img , IMG_SIZE)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5), 0)

    return img , gray , blur









if __name__ == "__main__" :
    img,gray,blur = load_and_preprocess(Path("data/images.jpg"))
    cv2.imshow('img',img)
    cv2.imshow('gray',gray)
    cv2.imshow('blur',blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(img.shape)