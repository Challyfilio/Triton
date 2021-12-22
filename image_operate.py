import cv2
import glob

i = 0
if __name__ == '__main__':
    for jpgfile in glob.glob(r'E:\ai_datasets\data\Tumor\Training\no_tumor\*.jpg'):
        image = cv2.imread(jpgfile)
        dst = cv2.flip(image, 1)  # 水平
        # cv2.imshow('image', dst)
        # cv2.waitKey()
        cv2.imwrite('E:\ai_datasets\data\Tumor\Training\out\out{}.jpg'.format(i), dst)
        i = i + 1
