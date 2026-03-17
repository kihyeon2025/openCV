import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./data/Lena.png', help='Image path.')
params = parser.parse_args()

img = cv2.imread(params.path)
print('original image shape:', img.shape)

width, height = 128, 256
resized_img = cv2.resize(img, (width, height))
print('resized to 128x256 image shape:', resized_img.shape)

w_mult, h_mult = 0.25, 0.5
resized_img = cv2.resize(img, (0, 0), resized_img, w_mult, h_mult)
print('image shape:', resized_img.shape)

w_mult, h_mult = 2, 4
resized_img = cv2.resize(img, (0, 0), resized_img, w_mult, h_mult, cv2.INTER_NEAREST)
print('image shape:', resized_img.shape)

img_flip_along_x = cv2.flip(img, 0)
img_flip_along_x_along_y = cv2.flip(img_flip_along_x, 1)
img_flipped_xy = cv2.flip(img, -1)

# check that sequential flips around x and y equal to simultaneous x-y flip
assert img_flipped_xy.all() == img_flip_along_x_along_y.all()
cv2.imshow("original", img)
cv2.imshow("resized", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./data/Lena.png', help='Image path.')
parser.add_argument('--out_png', default='./data/Lena_compressed.png',
                    help='Output image path for lossless result.')
parser.add_argument('--out_jpg', default='./data/Lena_compressed.jpg',
                    help='Output image path for lossy result.')

params = parser.parse_args()

img = cv2.imread(params.path)

# save image with lower compression - bigger file size but faster decoding
cv2.imwrite(params.out_png, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# check that image saved and loaded again image is the same as original one
saved_img = cv2.imread(params.out_png)
assert saved_img.all() == img.all()

# save image with lower quality - smaller file size
cv2.imwrite(params.out_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 0])