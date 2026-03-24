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

# 모폴로지 연산을 위한 커널 생성
MORPH_KERNEL_SIZE = 5
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

# 트랙바 콜백 함수
def on_threshold_changed(value):
    # Threshold 적용
    _, sobel_binary = cv2.threshold(sobel_result, value, 255, cv2.THRESH_BINARY)
    _, gabor_binary = cv2.threshold(gabor_combined, value, 255, cv2.THRESH_BINARY)
    
    # Opening 적용 (Erosion → Dilation) - 작은 노이즈 제거
    sobel_opened = cv2.morphologyEx(sobel_binary, cv2.MORPH_OPEN, morph_kernel)
    gabor_opened = cv2.morphologyEx(gabor_binary, cv2.MORPH_OPEN, morph_kernel)
    
    # Closing 적용 (Dilation → Erosion) - 구멍 채우기
    sobel_closed = cv2.morphologyEx(sobel_opened, cv2.MORPH_CLOSE, morph_kernel)
    gabor_closed = cv2.morphologyEx(gabor_opened, cv2.MORPH_CLOSE, morph_kernel)
    
    # 2x5 레이아웃으로 결과 표시
    display = np.zeros((gray.shape[0] * 2, gray.shape[1] * 5, 3), dtype=np.uint8)
    
    # 1행: 원본, Sobel(Threshold), Sobel(Opening), Sobel(Closing), 차이1
    display[0:gray.shape[0], 0:gray.shape[1]] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display[0:gray.shape[0], gray.shape[1]:gray.shape[1]*2] = cv2.cvtColor(sobel_binary, cv2.COLOR_GRAY2BGR)
    display[0:gray.shape[0], gray.shape[1]*2:gray.shape[1]*3] = cv2.cvtColor(sobel_opened, cv2.COLOR_GRAY2BGR)
    display[0:gray.shape[0], gray.shape[1]*3:gray.shape[1]*4] = cv2.cvtColor(sobel_closed, cv2.COLOR_GRAY2BGR)
    
    # 2행: 원본, Gabor(Threshold), Gabor(Opening), Gabor(Closing), 차이2
    display[gray.shape[0]:gray.shape[0]*2, 0:gray.shape[1]] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display[gray.shape[0]:gray.shape[0]*2, gray.shape[1]:gray.shape[1]*2] = cv2.cvtColor(gabor_binary, cv2.COLOR_GRAY2BGR)
    display[gray.shape[0]:gray.shape[0]*2, gray.shape[1]*2:gray.shape[1]*3] = cv2.cvtColor(gabor_opened, cv2.COLOR_GRAY2BGR)
    display[gray.shape[0]:gray.shape[0]*2, gray.shape[1]*3:gray.shape[1]*4] = cv2.cvtColor(gabor_closed, cv2.COLOR_GRAY2BGR)
    
    # 차이 표시 (마지막 열)
    diff_before = cv2.absdiff(sobel_binary, gabor_binary)
    diff_after = cv2.absdiff(sobel_closed, gabor_closed)
    
    diff_before_colored = cv2.cvtColor(diff_before, cv2.COLOR_GRAY2BGR)
    diff_before_colored[:, :, 2] = diff_before  # Red 채널
    display[0:gray.shape[0], gray.shape[1]*4:gray.shape[1]*5] = diff_before_colored
    
    diff_after_colored = cv2.cvtColor(diff_after, cv2.COLOR_GRAY2BGR)
    diff_after_colored[:, :, 2] = diff_after  # Red 채널
    display[gray.shape[0]:gray.shape[0]*2, gray.shape[1]*4:gray.shape[1]*5] = diff_after_colored
    
    # 통계 정보
    sobel_binary_pixels = np.count_nonzero(sobel_binary)
    sobel_closed_pixels = np.count_nonzero(sobel_closed)
    gabor_binary_pixels = np.count_nonzero(gabor_binary)
    gabor_closed_pixels = np.count_nonzero(gabor_closed)
    diff_before_pixels = np.count_nonzero(diff_before)
    diff_after_pixels = np.count_nonzero(diff_after)
    
    info_text = f"Threshold: {value} | Kernel: {MORPH_KERNEL_SIZE}x{MORPH_KERNEL_SIZE}"
    cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    stats_text = f"Before: Sobel={sobel_binary_pixels} Gabor={gabor_binary_pixels} Diff={diff_before_pixels}"
    cv2.putText(display, stats_text, (10, gray.shape[0] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    stats_text2 = f"After: Sobel={sobel_closed_pixels} Gabor={gabor_closed_pixels} Diff={diff_after_pixels}"
    cv2.putText(display, stats_text2, (10, gray.shape[0] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # 라벨 추가
    label_y = [gray.shape[0] // 2 - 10, gray.shape[0] + gray.shape[0] // 2 - 10]
    labels = ["Sobel", "Gabor"]
    for idx, label_text in enumerate(labels):
        cv2.putText(display, label_text, (10, label_y[idx]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Morphological Operations - Opening & Closing', display)

# 윈도우 생성
window_name = 'Morphological Operations - Opening & Closing'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1600, 600)

# 트랙바 생성
cv2.createTrackbar('Threshold', window_name, 127, 255, on_threshold_changed)

print("=" * 70)
print("Morphological Operations 비교: Opening & Closing")
print("=" * 70)
print("1행: Sobel Filter")
print("  - 원본 | Threshold | Opening | Closing | Diff(Before/After)")
print("")
print("2행: Gabor Filter")
print("  - 원본 | Threshold | Opening | Closing | Diff(Before/After)")
print("")
print("Opening (침식→팽창): 작은 노이즈 제거")
print("Closing (팽창→침식): 객체 내부의 구멍 채우기")
print("")
print("트랙바를 조절하여 Threshold를 변경하세요.")
print("아무 키나 눌러서 종료하세요.")
print("=" * 70)

# 초기 표시
on_threshold_changed(127)

cv2.waitKey(0)
cv2.destroyAllWindows()

