import cv2
import speech_recognition as sr
import pyttsx3
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO(r"C:\Rosie\YOLO\Object-Detection\Object-Detection\Yolo-Weights\yolov8l.pt")

# 클래스 이름 매핑
CLASS_MAP = {
    "사람": "person", "자전거": "bicycle", "자동차": "car", "오토바이": "motorbike",
    "비행기": "aeroplane", "버스": "bus", "기차": "train", "트럭": "truck", "배": "boat",
    "신호등": "traffic light", "소화전": "fire hydrant", "정지신호": "stop sign", 
    "주차료 징수기": "parking meter", "벤치": "bench", "새": "bird", "고양이": "cat", 
    "개": "dog", "말": "horse", "양": "sheep", "소": "cow", "코끼리": "elephant", 
    "곰": "bear", "얼룩말": "zebra", "기린": "giraffe", "가방": "backpack", "우산": "umbrella",
    "핸드백": "handbag", "넥타이": "tie", "서류가방": "suitcase", "프리스비": "frisbee", 
    "스키": "skis", "스노우보드": "snowboard", "스포츠볼": "sports ball", "연": "kite",
    "야구방망이": "baseball bat", "야구글러브": "baseball glove", "스케이트보드": "skateboard",
    "서핑보드": "surfboard", "테니스라켓": "tennis racket", "물병": "bottle", 
    "와인잔": "wine glass", "커프": "cup", "포크": "fork", "나이프": "knife", 
    "스푼": "spoon", "보울": "bowl", "바나나": "banana", "사과": "apple", 
    "샌드위치": "sandwich", "오렌지": "orange", "브로콜리": "broccoli", 
    "당근": "carrot", "핫도그": "hot dog", "피자": "pizza", "도넛": "donut", 
    "케이크": "cake", "의자": "chair", "소파": "sofa", "화분": "pottedplant", 
    "침대": "bed", "테이블": "diningtable", "화장실": "toilet", "모니터": "tvmonitor", 
    "노트북": "laptop", "마우스": "mouse", "리모컨": "remote", "키보드": "keyboard", 
    "핸드폰": "cell phone", "전자렌지": "microwave", "오븐": "oven", 
    "토스터기": "toaster", "싱크대": "sink", "냉장고": "refrigerator", "책": "book",
    "시계": "clock", "화병": "vase", "가위": "scissors", "곰인형": "teddy bear",
    "드라이기": "hair drier", "칫솔": "toothbrush",
}

# 음성 안내 초기화
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 음성 속도 설정

# 음성 인식 함수
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # "말씀하세요" 음성 안내
        engine.say("말씀하세요")
        engine.runAndWait()
        print("말씀하세요...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="ko-KR")
        print(f"인식된 텍스트: {text}")
        return text
    except sr.UnknownValueError:
        print("음성을 인식하지 못했습니다.")
        return None
    except sr.RequestError as e:
        print(f"음성 인식 서비스 오류: {e}")
        return None

# 카메라로 객체 탐지
def detect_object(target_class):
    cap = cv2.VideoCapture(0)  # 카메라 열기

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        engine.say("카메라를 열 수 없습니다.")
        engine.runAndWait()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # YOLO 탐지 수행
        results = model.predict(source=frame, show=True, conf=0.5)  # 화면에 결과 표시
        detections = results[0].boxes.data.cpu().numpy()  # 탐지된 객체 정보

        found = False
        for det in detections:
            class_id = int(det[5])  # 클래스 ID
            confidence = det[4]    # 신뢰도
            x1, y1, x2, y2 = map(int, det[:4])  # 바운딩 박스 좌표

            if model.names[class_id] == target_class:
                found = True
                # 객체 위치 계산 (화면 기준)
                width, height = frame.shape[1], frame.shape[0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                position_x = "왼쪽" if center_x < width / 3 else "오른쪽" if center_x > 2 * width / 3 else "가운데"
                position_y = "위" if center_y < height / 3 else "아래" if center_y > 2 * height / 3 else "중간"

                # 음성 안내
                message = f"{target_class}이 화면 {position_y} {position_x}에 있습니다."
                print(message)
                engine.say(message)
                engine.runAndWait()
                break

        if found:
            break

    cap.release()
    cv2.destroyAllWindows()

# 메인 함수
if __name__ == "__main__":
    while True:
        target = recognize_speech()
        if target is None:
            continue

        # 클래스 매핑
        if target in CLASS_MAP:
            target_class = CLASS_MAP[target]
            detect_object(target_class)
        else:
            print(f"{target}은(는) 지원하지 않는 객체입니다.")
            engine.say(f"{target}은 지원하지 않는 객체입니다.")
            engine.runAndWait()
