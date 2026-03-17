import cv2
import numpy as np
# 1. 이미지 읽기 (컬러)
image = cv2.imread('./data/Lena.png', cv2.IMREAD_COLOR)

# 2. 원본 이미지 화면 출력
cv2.imshow('Lena Color Image', image)

# 3. 흑백 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray)
cv2.moveWindow('Gray Image', 100, 100)

# 4. 히스토그램 적용
equalized = cv2.equalizeHist(gray)
cv2.imshow('Histogram Equalized', equalized)

gamma = 3.0
# 5. 감마 보정 적용
gamma_corrected = np.power(gray / 255.0, gamma) * 255
gamma_corrected = gamma_corrected.astype(np.uint8)
cv2.imshow('Gamma Corrected', gamma_corrected)


# 키 입력 대기
cv2.waitKey(0)

# 창 닫기
cv2.destroyAllWindows()