import cv2
import numpy as np

# 1. 원본 이미지 읽기
image = cv2.imread('./data/Lena.png')

# 2. HSV 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 채널 분리
h, s, v = cv2.split(hsv)

# 4. 0~255 정규화
h_norm = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
s_norm = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)
v_norm = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

# 5. 각각 출력
cv2.imshow('H channel', h_norm)
cv2.imshow('S channel', s_norm)
cv2.imshow('V channel', v_norm)

# 6. 필터 적용
h_median = cv2.medianBlur(h, 5)
s_gaussian = cv2.GaussianBlur(s, (5, 5), 0)
v_bilateral = cv2.bilateralFilter(v, 9, 75, 75)

# 7. 필터 결과 출력
cv2.imshow('H Median Filter', h_median)
cv2.imshow('S Gaussian Filter', s_gaussian)
cv2.imshow('V Bilateral Filter', v_bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()