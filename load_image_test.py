import numpy as np
import skimage.io
import skimage.transform    
import cv2

def load_image_1(path, size=224):
    image = skimage.io.imread(path)
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    image = image[yy:yy + short_edge, xx:xx + short_edge]
    image = skimage.transform.resize(image, (size, size))
    return image


def load_image_2(fname):
    image = cv2.imread(fname).astype('float32') / 255
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    image = image[yy:yy + short_edge, xx:xx + short_edge]
    image = cv2.resize(image, (224, 224))
    image = image[:, :, [2, 1, 0]]
    return image

def main():
    fname = 'data/cat.jpg'
    img1 = load_image_1(fname)
    img2 = load_image_2(fname)
    print '1', img1[:5, 0, :]
    print '2', img2[:5, 0, :]
    q = img1 / img2
    print 'q', q[:5, 0, :]

if __name__ == '__main__':
    main()
