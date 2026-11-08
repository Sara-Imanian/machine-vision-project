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


def segment_leaf(blur):
    _,mask = cv2.threshold(blur ,150, 200 , cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours,_ = cv2.findContours(mask,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    leaf_contour = max(contours , key=cv2.contourArea)

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [leaf_contour], -1 , 255, thickness=cv2.FILLED)

    return clean_mask , leaf_contour


def extract_shape_features(contour):
    area = cv2.contourArea(contour)
    permiter = cv2.arcLength(contour , True)

    x , y , w, h = cv2.boundingRect(contour)
    aspect_ratio = w/h
    extent = area/(w*h)

    hull = cv2.convexHull(contour)
    hul_area = cv2.contourArea(hull)
    solidity = area/hul_area
    circularity = 4 * np.pi * area/(permiter ** 2)

    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    features = np.array([aspect_ratio, extent, solidity, circularity, *hu])
    return features


def extract_color_features(img, mask, bins=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0,180])
    s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0,256])
    v_hist = cv2.calcHist([hsv], [2], mask, [bins], [0,256])

    hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()
    hist = hist / (hist.sum() + 1e-12)

    return hist


def extract_texture_features(gray, mask):
    mean, std = cv2.meanStdDev(gray, mask=mask)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap[mask > 0].var()

    return np.array([mean[0][0], std[0][0], lap_var])


def extract_all_features(img, gray, mask, contour): 
    shape = extract_shape_features(contour)
    color = extract_color_features(img, mask)
    texture = extract_texture_features(gray, mask)

    return np.concatenate([shape, color, texture])



if __name__ == "__main__" :

    img,gray,blur = load_and_preprocess(Path("data/images.jpg"))

    # cv2.imshow('img',img)
    # cv2.imshow('gray',gray)
    # cv2.imshow('blur',blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    mask , contour = segment_leaf(blur)

    result = img.copy()
    cv2.drawContours(result,[contour], -1, (0, 255,0), 2)
    cv2.imshow("mask", mask)
    cv2.imshow('result',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    features = extract_shape_features(contour)

    features = extract_all_features(img, gray, mask, contour)

np.set_printoptions(precision=3, suppress=True)
print(features.shape)

# print(img.shape)