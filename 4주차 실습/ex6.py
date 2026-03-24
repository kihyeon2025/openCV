import cv2
import numpy as np

# 비디오 파일 경로
video_path = '../data/samples_data_vtest.avi'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 비디오 정보 확인
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"비디오 정보: {width}x{height}, FPS: {fps}, 총 프레임: {frame_count}")

# DFT 및 필터링을 위한 함수 정의
def compute_dft(image):
    """이미지에 DFT 적용"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift

def create_filter(shape, filter_type, cutoff):
    """주파수 도메인 필터 생성"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # 거리 행렬 생성
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    if filter_type == 0:  # Low-pass filter
        mask = np.where(distance <= cutoff, 1, 0)
    elif filter_type == 1:  # High-pass filter
        mask = np.where(distance > cutoff, 1, 0)
    elif filter_type == 2:  # Band-pass filter (간단한 버전)
        mask = np.where((distance > cutoff/2) & (distance <= cutoff), 1, 0)
    else:  # No filter
        mask = np.ones((rows, cols))
    
    return mask.astype(np.float32)

def apply_filter(dft_shift, filter_mask):
    """필터 적용"""
    filtered_dft = dft_shift * filter_mask[:, :, np.newaxis]
    return filtered_dft

def inverse_dft(filtered_dft, original_shape):
    """역 DFT 적용"""
    dft_ishift = np.fft.ifftshift(filtered_dft)
    img_back = cv2.idft(dft_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img_back

# 전역 변수
current_frame = None
dft_result = None
filtered_result = None
reconstructed = None
filter_type = 0  # 0: LPF, 1: HPF, 2: BPF, 3: None
cutoff_value = 30

def update_display():
    """화면 업데이트"""
    global current_frame, dft_result, filtered_result, reconstructed
    
    if current_frame is None:
        return
    
    # DFT 계산
    dft_shift = compute_dft(current_frame)
    
    # Magnitude 계산 (시각화용)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude = np.log(magnitude + 1)
    dft_result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 필터 생성 및 적용
    filter_mask = create_filter(dft_shift.shape[:2], filter_type, cutoff_value)
    filtered_dft = apply_filter(dft_shift, filter_mask)
    
    # 필터링된 magnitude
    filtered_magnitude = cv2.magnitude(filtered_dft[:, :, 0], filtered_dft[:, :, 1])
    filtered_magnitude = np.log(filtered_magnitude + 1)
    filtered_result = cv2.normalize(filtered_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 역 DFT로 재구성
    reconstructed = inverse_dft(filtered_dft, current_frame.shape[:2])
    
    # 2x2 레이아웃으로 표시
    display = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    
    # 좌상단: 원본 프레임
    display[0:height, 0:width] = current_frame
    
    # 우상단: DFT Magnitude
    display[0:height, width:width*2] = cv2.cvtColor(dft_result, cv2.COLOR_GRAY2BGR)
    
    # 좌하단: 필터링된 DFT Magnitude
    display[height:height*2, 0:width] = cv2.cvtColor(filtered_result, cv2.COLOR_GRAY2BGR)
    
    # 우하단: 재구성된 이미지
    display[height:height*2, width:width*2] = cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2BGR)
    
    # 정보 표시
    filter_names = ["Low-Pass", "High-Pass", "Band-Pass", "No Filter"]
    info_text = f"Filter: {filter_names[filter_type]} | Cutoff: {cutoff_value}"
    cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 라벨 추가
    cv2.putText(display, "Original", (10, height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "DFT Magnitude", (width + 10, height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "Filtered DFT", (10, height*2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "Reconstructed", (width + 10, height*2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Frequency-based Filtering', display)

def on_filter_type_changed(value):
    """필터 타입 변경 콜백"""
    global filter_type
    filter_type = value
    update_display()

def on_cutoff_changed(value):
    """Cutoff 값 변경 콜백"""
    global cutoff_value
    cutoff_value = value
    update_display()

# 윈도우 생성
window_name = 'Frequency-based Filtering'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 800)

# 트랙바 생성
cv2.createTrackbar('Filter Type', window_name, 0, 3, on_filter_type_changed)
cv2.createTrackbar('Cutoff', window_name, 30, 100, on_cutoff_changed)

print("=" * 70)
print("Frequency-based Filtering")
print("=" * 70)
print("Filter Types:")
print("  0: Low-Pass Filter (저주파 통과)")
print("  1: High-Pass Filter (고주파 통과)")
print("  2: Band-Pass Filter (대역 통과)")
print("  3: No Filter (필터 없음)")
print("")
print("트랙바를 사용하여 필터 타입과 Cutoff 값을 조절하세요.")
print("'q' 키를 눌러서 종료하세요.")
print("=" * 70)

frame_idx = 0

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("비디오 끝에 도달했습니다.")
        break
    
    frame_idx += 1
    current_frame = frame
    
    # 화면 업데이트
    update_display()
    
    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()

print("비디오 분석 완료!")
