import cv2
import numpy as np

# Lena.png 파일 읽기
image = cv2.imread('../data/Lena.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ========== Sobel Filter 적용 (ex2) ==========
KSIZE = 3

# X 방향 미분 계산
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=KSIZE)

# Y 방향 미분 계산
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=KSIZE)

# Sobel 크기 계산 (magnitude)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# 0-255 범위로 정규화
sobel_result = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

# ========== Gabor Filter 적용 (ex3) ==========
KSIZE_GABOR = 21
SIGMA = 3.0
LAMBD = 10.0
GAMMA = 0.5
PSI = 0

angles = [0, 45, 90, 135]
gabor_results = []

for angle in angles:
    theta = np.pi * angle / 180.0
    kernel = cv2.getGaborKernel((KSIZE_GABOR, KSIZE_GABOR), SIGMA, theta, LAMBD, GAMMA, PSI)
    kernel = kernel / np.sum(np.abs(kernel))
    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
    filtered = np.uint8(255 * (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-6))
    gabor_results.append(filtered)

# Gabor 결과 합치기 (모든 방향의 평균)
gabor_combined = np.mean(gabor_results, axis=0).astype(np.uint8)

# 트랙바 콜백 함수
def on_threshold_changed(value):
    # Threshold 적용
    _, sobel_binary = cv2.threshold(sobel_result, value, 255, cv2.THRESH_BINARY)
    _, gabor_binary = cv2.threshold(gabor_combined, value, 255, cv2.THRESH_BINARY)
    
    # 두 필터 결과의 차이 계산
    difference = cv2.absdiff(sobel_binary, gabor_binary)
    
    # 결과 표시
    # 상단: 원본 이미지
    # 중단: Sobel vs Gabor (Threshold 적용)
    # 하단: 차이
    
    display = np.zeros((gray.shape[0] * 3, gray.shape[1] * 2, 3), dtype=np.uint8)
    
    # 원본 이미지
    display[0:gray.shape[0], 0:gray.shape[1]] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display[0:gray.shape[0], gray.shape[1]:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Sobel 필터 결과 (Threshold 적용)
    display[gray.shape[0]:gray.shape[0]*2, 0:gray.shape[1]] = cv2.cvtColor(sobel_binary, cv2.COLOR_GRAY2BGR)
    
    # Gabor 필터 결과 (Threshold 적용)
    display[gray.shape[0]:gray.shape[0]*2, gray.shape[1]:] = cv2.cvtColor(gabor_binary, cv2.COLOR_GRAY2BGR)
    
    # 차이 (빨간색으로 표시)
    difference_colored = cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)
    difference_colored[:, :, 2] = difference  # Red 채널에 차이 표시
    display[gray.shape[0]*2:, 0:gray.shape[1]] = difference_colored
    
    # 통계 정보
    sobel_pixels = np.count_nonzero(sobel_binary)
    gabor_pixels = np.count_nonzero(gabor_binary)
    difference_pixels = np.count_nonzero(difference)
    
    info_text = f"Threshold: {value} | Sobel: {sobel_pixels} | Gabor: {gabor_pixels} | Diff: {difference_pixels}"
    cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 라벨 추가
    cv2.putText(display, "Original", (10, gray.shape[0] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "Sobel (Threshold)", (10, gray.shape[0] * 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "Gabor (Threshold)", (gray.shape[1] + 10, gray.shape[0] * 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "Difference", (10, gray.shape[0] * 3 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Sobel vs Gabor - Threshold Comparison', display)

# 윈도우 생성
window_name = 'Sobel vs Gabor - Threshold Comparison'
cv2.namedWindow(window_name)

# 트랙바 생성 (0 ~ 255 범위)
cv2.createTrackbar('Threshold', window_name, 127, 255, on_threshold_changed)

print("트랙바를 조절하여 Threshold를 변경하세요.")
print("Sobel과 Gabor 필터 결과의 차이를 확인할 수 있습니다.")
print("아무 키나 눌러서 종료하세요.")

# 초기 표시
on_threshold_changed(127)

cv2.waitKey(0)
cv2.destroyAllWindows()

