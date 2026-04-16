import cv2
import math
import numpy as np
from ultralytics import YOLO
import torch

# ─── COCO keypoint indices ────────────────────────────────────────────────────
_KP_NOSE        = 0
_KP_L_SHOULDER  = 5
_KP_R_SHOULDER  = 6
_KP_L_HIP       = 11
_KP_R_HIP       = 12


def _body_orientation(kps: np.ndarray) -> list | None:
    """
    YOLO pose keypoints (17, 3) → [ox, oy, dx, dy]

    계산법:
      shoulder_mid = (l_shoulder + r_shoulder) / 2
      shoulder_vec = r_shoulder - l_shoulder
      normal       = perpendicular(shoulder_vec) 방향이 nose 쪽
      ori = [shoulder_mid.x, shoulder_mid.y, normal.x, normal.y]

    반환값: [ox, oy, dx, dy] float 4개 또는 None
    """
    if kps is None or kps.shape[0] < 13:
        return None

    CONF = 0.3  # 신뢰도 임계값

    def pt(idx):
        """유효한 키포인트 반환, 없으면 None"""
        if kps[idx, 2] >= CONF:
            return kps[idx, :2]
        return None

    ls, rs = pt(_KP_L_SHOULDER), pt(_KP_R_SHOULDER)
    if ls is None or rs is None:
        return None

    # 어깨 중점 (origin)
    ox = (ls[0] + rs[0]) / 2.0
    oy = (ls[1] + rs[1]) / 2.0

    # 어깨 벡터 (오른쪽 → 왼쪽)
    svx = ls[0] - rs[0]
    svy = ls[1] - rs[1]

    # 수직(perpendicular): (-svy, svx) 또는 (svy, -svx) — nose 방향으로 선택
    perp1 = np.array([-svy,  svx])
    perp2 = np.array([ svy, -svx])

    nose = pt(_KP_NOSE)
    if nose is not None:
        # nose 쪽을 향하는 수직 선택
        to_nose = nose - np.array([ox, oy])
        normal = perp1 if np.dot(perp1, to_nose) >= 0 else perp2
    else:
        # nose 없으면 위쪽(y 감소) 방향 기본값
        normal = perp1 if perp1[1] <= 0 else perp2

    # 정규화
    length = math.sqrt(normal[0]**2 + normal[1]**2)
    if length < 1e-6:
        return None

    dx, dy = normal[0] / length, normal[1] / length
    return [round(float(ox), 2), round(float(oy), 2),
            round(float(dx), 4), round(float(dy), 4)]


class YOLODetector:
    def __init__(self, model_path='yolov8n-pose.pt', device='mps'):
        """
        Initialize YOLO pose detector
        pose 모델 사용 시 keypoints → ori 자동 계산
        bbox 전용 모델도 하위 호환 지원

        Args:
            model_path: YOLO 모델 경로 (pose 권장: yolov8n-pose.pt)
            device: 'mps' | 'cuda' | 'cpu'
        """
        self.device = device
        self.model = YOLO(model_path)
        self._is_pose = 'pose' in str(model_path).lower()
        
        # Move model to specified device
        if device == 'mps' and torch.backends.mps.is_available():
            self.model.model = self.model.model.to('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            self.model.model = self.model.model.to('cuda')
        else:
            self.model.model = self.model.model.to('cpu')

        # Classes we care about for hockey
        self.target_classes = [0]  # person class

    def detect(self, frame):
        """
        Detect persons in frame. pose 모델일 때 keypoints + ori 포함.

        Returns:
            List[dict] with keys:
              - bbox: [x1, y1, x2, y2]
              - confidence: float
              - class_id: int
              - class_name: str
              - keypoints: [[x,y,conf]×17]  (pose 모델 전용, 없으면 키 없음)
              - ori: [ox, oy, dx, dy]       (pose 모델 + 어깨 신뢰도 충분 시)
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            # pose 모델: keypoints 텐서 (N, 17, 3)
            kps_all = None
            if self._is_pose and result.keypoints is not None:
                kps_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id   = int(box.cls[0].cpu().numpy())

                if class_id not in self.target_classes or confidence <= 0.3:
                    continue

                det = {
                    'bbox':       [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id':   class_id,
                    'class_name': self.model.names[class_id],
                }

                # pose 키포인트 & 방향 벡터
                if kps_all is not None and i < len(kps_all):
                    kp = kps_all[i]  # (17, 3)
                    det['keypoints'] = kp.tolist()
                    ori = _body_orientation(kp)
                    if ori:
                        det['ori'] = ori

                detections.append(det)

        return detections
    
    def detect_batch(self, frames):
        """
        Detect objects in batch of frames
        
        Args:
            frames: List of frames
            
        Returns:
            batch_detections: List of detection lists for each frame
        """
        batch_detections = []
        
        for frame in frames:
            detections = self.detect(frame)
            batch_detections.append(detections)
            
        return batch_detections
    
    def visualize_detections(self, frame, detections, thickness=2):
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            thickness: Box line thickness
            
        Returns:
            annotated_frame: Frame with detection boxes
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness)
        
        return annotated_frame

class PuckDetector:
    def __init__(self, device='mps'):
        """
        Initialize puck detector using color/motion detection
        
        Args:
            device: Device for any ML operations
        """
        self.device = device
        self.prev_frame = None
        
        # Puck detection parameters
        self.min_area = 10
        self.max_area = 200
        self.min_circularity = 0.3
        
    def detect_puck(self, frame):
        """
        Detect puck in frame using color and motion
        
        Args:
            frame: Input frame
            
        Returns:
            puck_detections: List of puck detection dictionaries
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to find dark objects (puck is typically dark)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        puck_detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_area <= area <= self.max_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > self.min_circularity:
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        detection = {
                            'bbox': [x, y, x + w, y + h],
                            'confidence': min(circularity * 2, 1.0),  # Use circularity as confidence
                            'class_id': 99,  # Special ID for puck
                            'class_name': 'puck',
                            'area': area,
                            'circularity': circularity
                        }
                        puck_detections.append(detection)
        
        # Sort by confidence and keep top detections
        puck_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.prev_frame = gray
        
        return puck_detections[:3]  # Return top 3 candidates
    
    def visualize_puck_detections(self, frame, detections, thickness=2):
        """
        Draw puck detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of puck detections
            thickness: Box line thickness
            
        Returns:
            annotated_frame: Frame with puck detection boxes
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box in red for puck
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
            
            # Draw label
            label = f"Puck: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 255), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness)
        
        return annotated_frame

# Legacy compatibility
HockeyDetector = YOLODetector
