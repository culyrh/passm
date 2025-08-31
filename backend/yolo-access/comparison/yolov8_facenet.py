# YOLOv8-Face + FaceNet 얼굴 인식 시스템

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class InceptionResNetV1(nn.Module):
    """FaceNet 모델 (PyTorch 구현)"""
    
    def __init__(self, pretrained=True, classify=False, num_classes=None):
        super(InceptionResNetV1, self).__init__()
        self.classify = classify
        self.num_classes = num_classes
        
        # 간단한 CNN 구조로 대체 (실제로는 복잡한 InceptionResNet 구조)
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 512)  # 512차원 임베딩
        
        if classify:
            self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if self.classify:
            x = self.classifier(x)
        else:
            x = F.normalize(x, p=2, dim=1)  # L2 정규화
        
        return x

class YOLOv8FaceNet:
    """YOLOv8-Face + FaceNet 하이브리드 시스템"""
    
    def __init__(self, 
                 yolo_model_path: str = "yolov8n-face.pt",
                 facenet_model_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Args:
            yolo_model_path: YOLOv8-Face 모델 경로
            facenet_model_path: FaceNet 모델 경로 (None이면 기본 모델)
            device: 연산 장치 ('cpu' 또는 'cuda')
        """
        self.device = device
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # 모델 로드
        self.load_yolo_model(yolo_model_path)
        self.load_facenet_model(facenet_model_path)
        
        # 사용자 데이터
        self.user_embeddings = {}  # {user_name: [embedding_list]}
        self.user_names = []
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info("✅ YOLOv8-Face + FaceNet 시스템 초기화 완료")
    
    def load_yolo_model(self, model_path: str):
        """YOLOv8-Face 모델 로드"""
        try:
            logger.info(f"📁 YOLOv8-Face 모델 로드 중: {model_path}")
            
            if not os.path.exists(model_path):
                logger.warning(f"모델 파일 없음. 기본 YOLOv8n 다운로드: {model_path}")
                # YOLOv8-Face 모델이 없으면 일반 YOLOv8으로 대체
                self.yolo_model = YOLO("yolov8n.pt")
            else:
                self.yolo_model = YOLO(model_path)
            
            self.yolo_model.to(self.device)
            
            # 워밍업
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.yolo_model(dummy_img, verbose=False)
            
            logger.info("✅ YOLOv8-Face 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ YOLOv8 모델 로드 실패: {e}")
            self.yolo_model = None
    
    def load_facenet_model(self, model_path: Optional[str]):
        """FaceNet 모델 로드"""
        try:
            logger.info("📁 FaceNet 모델 로드 중...")
            
            self.facenet_model = InceptionResNetV1(pretrained=True, classify=False)
            self.facenet_model.to(self.device)
            self.facenet_model.eval()
            
            # 사전 훈련된 가중치가 있다면 로드
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.facenet_model.load_state_dict(checkpoint)
                logger.info(f"✅ 사전 훈련된 FaceNet 가중치 로드: {model_path}")
            else:
                logger.warning("⚠️ 사전 훈련된 가중치 없음. 랜덤 초기화된 모델 사용")
            
            logger.info("✅ FaceNet 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ FaceNet 모델 로드 실패: {e}")
            self.facenet_model = None
    
    def detect_faces(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """YOLOv8-Face로 얼굴 검출"""
        start_time = time.time()
        
        if self.yolo_model is None:
            return [], 0.0
        
        try:
            # YOLO 추론
            results = self.yolo_model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
            
            faces = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 사람/얼굴 클래스만 필터링
                        class_name = self.yolo_model.names.get(class_id, "unknown")
                        if class_name.lower() in ['person', 'face', '0']:
                            
                            # 바운딩 박스 좌표 정리
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w, h = x2 - x1, y2 - y1
                            
                            # 이미지 경계 체크
                            img_h, img_w = image.shape[:2]
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(img_w, x2)
                            y2 = min(img_h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                face_info = {
                                    'bbox': [x1, y1, x2-x1, y2-y1],
                                    'confidence': confidence,
                                    'area': (x2-x1) * (y2-y1),
                                    'method': 'yolov8'
                                }
                                faces.append(face_info)
            
            # 면적 기준 정렬 (큰 얼굴 우선)
            faces.sort(key=lambda x: x['area'], reverse=True)
            
            processing_time = time.time() - start_time
            return faces, float(processing_time)
            
        except Exception as e:
            logger.error(f"❌ YOLO 얼굴 검출 오류: {e}")
            return [], float(time.time() - start_time)
    
    def extract_face_embedding(self, image: np.ndarray, face_bbox: List[int]) -> np.ndarray:
        """FaceNet으로 얼굴 임베딩 추출"""
        if self.facenet_model is None:
            return np.array([])
        
        try:
            x, y, w, h = face_bbox
            
            # 얼굴 영역 추출
            face_region = image[y:y+h, x:x+w]
            if face_region.size == 0:
                return np.array([])
            
            # BGR -> RGB 변환
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # 전처리
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # FaceNet 임베딩 추출
            with torch.no_grad():
                embedding = self.facenet_model(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ FaceNet 임베딩 추출 오류: {e}")
            return np.array([])
    
    def load_user_database(self, users_dir: str = "data/users"):
        """사용자 얼굴 데이터베이스 로드"""
        try:
            if not os.path.exists(users_dir):
                logger.warning(f"사용자 데이터 폴더 없음: {users_dir}")
                return
            
            self.user_embeddings = {}
            self.user_names = []
            
            for user_name in os.listdir(users_dir):
                user_path = os.path.join(users_dir, user_name)
                
                if os.path.isdir(user_path):
                    logger.info(f"📄 {user_name} 데이터 로드 중...")
                    
                    user_embeddings_list = []
                    processed_count = 0
                    
                    for image_file in os.listdir(user_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(user_path, image_file)
                            
                            try:
                                # 이미지 로드
                                image = cv2.imread(image_path)
                                if image is None:
                                    continue
                                
                                # 얼굴 검출
                                faces, _ = self.detect_faces(image)
                                
                                if faces:
                                    # 가장 큰 얼굴 선택
                                    largest_face = max(faces, key=lambda x: x['area'])
                                    
                                    # FaceNet 임베딩 추출
                                    embedding = self.extract_face_embedding(image, largest_face['bbox'])
                                    
                                    if len(embedding) > 0:
                                        user_embeddings_list.append(embedding)
                                        processed_count += 1
                                        
                            except Exception as e:
                                logger.warning(f"⚠️ {image_file} 처리 실패: {e}")
                                continue
                    
                    if user_embeddings_list:
                        self.user_embeddings[user_name] = user_embeddings_list
                        self.user_names.append(user_name)
                        logger.info(f"✅ {user_name}: {processed_count}장 처리 완료")
                    else:
                        logger.warning(f"⚠️ {user_name}: 처리 가능한 이미지 없음")
            
            logger.info(f"🎯 총 {len(self.user_names)}명의 사용자 데이터 로드 완료")
            
            # 임베딩 데이터 저장
            self.save_embeddings()
            
        except Exception as e:
            logger.error(f"❌ 사용자 데이터베이스 로드 실패: {e}")
    
    def save_embeddings(self, save_path: str = "data/facenet_embeddings.pkl"):
        """임베딩 데이터 저장"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'user_embeddings': self.user_embeddings,
                    'user_names': self.user_names
                }, f)
            logger.info(f"💾 임베딩 데이터 저장: {save_path}")
        except Exception as e:
            logger.error(f"❌ 임베딩 저장 실패: {e}")
    
    def load_embeddings(self, load_path: str = "data/facenet_embeddings.pkl"):
        """임베딩 데이터 로드"""
        try:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    data = pickle.load(f)
                self.user_embeddings = data['user_embeddings']
                self.user_names = data['user_names']
                logger.info(f"📂 임베딩 데이터 로드: {len(self.user_names)}명")
                return True
        except Exception as e:
            logger.error(f"❌ 임베딩 로드 실패: {e}")
        return False
    
    def recognize_user(self, face_embedding: np.ndarray, threshold: float = 0.6) -> Dict:
        """FaceNet 임베딩으로 사용자 인식"""
        if len(self.user_embeddings) == 0:
            return {
                'recognized': False,
                'user_name': '데이터베이스 없음',
                'confidence': 0.0,
                'distance': 999.0,
                'message': '등록된 사용자 데이터가 없습니다'
            }
        
        if len(face_embedding) == 0:
            return {
                'recognized': False,
                'user_name': '특징 추출 실패',
                'confidence': 0.0,
                'distance': 999.0,
                'message': '얼굴 특징을 추출할 수 없습니다'
            }
        
        try:
            best_distance = float('inf')
            best_user = None
            all_distances = {}
            
            # 각 사용자와의 거리 계산
            for user_name, embeddings_list in self.user_embeddings.items():
                user_distances = []
                
                for stored_embedding in embeddings_list:
                    # 유클리드 거리 계산
                    distance = np.linalg.norm(face_embedding - stored_embedding)
                    user_distances.append(distance)
                
                # 해당 사용자의 최소 거리 (가장 유사한 사진)
                min_distance = min(user_distances)
                all_distances[user_name] = min_distance
                
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_user = user_name
            
            # 신뢰도 계산 (거리를 백분율로 변환)
            confidence = max(0, (1 - best_distance) * 100)
            
            # 임계값 확인
            if best_distance < threshold and best_user is not None:
                return {
                    'recognized': True,
                    'user_name': str(best_user),
                    'confidence': float(round(confidence, 2)),
                    'distance': float(round(best_distance, 4)),
                    'threshold': float(threshold),
                    'all_distances': {k: float(round(v, 4)) for k, v in all_distances.items()},
                    'message': f'{best_user} 인식 성공'
                }
            else:
                return {
                    'recognized': False,
                    'user_name': '미등록 인물',
                    'confidence': float(round(confidence, 2)),
                    'distance': float(round(best_distance, 4)),
                    'threshold': float(threshold),
                    'best_match': str(best_user) if best_user else '매칭 없음',
                    'all_distances': {k: float(round(v, 4)) for k, v in all_distances.items()},
                    'message': f'거리 {round(best_distance, 4)} > 허용값 {threshold}'
                }
                
        except Exception as e:
            logger.error(f"❌ 사용자 인식 오류: {e}")
            return {
                'recognized': False,
                'user_name': '인식 오류',
                'confidence': 0.0,
                'distance': 999.0,
                'message': f'인식 처리 중 오류: {str(e)}'
            }
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        전체 이미지 처리 파이프라인
        1. YOLOv8-Face로 얼굴 검출
        2. FaceNet으로 임베딩 추출 및 인식
        """
        start_time = time.time()
        
        try:
            # 1. 얼굴 검출
            faces, detection_time = self.detect_faces(image)
            
            if not faces:
                return {
                    'success': False,
                    'faces_found': 0,
                    'message': '얼굴을 찾을 수 없습니다',
                    'performance': {
                        'detection_time': float(round(detection_time, 3)),
                        'total_time': float(round(time.time() - start_time, 3))
                    }
                }
            
            # 2. 가장 큰 얼굴 선택
            main_face = faces[0]
            
            # 3. FaceNet 임베딩 추출
            embedding_start = time.time()
            face_embedding = self.extract_face_embedding(image, main_face['bbox'])
            embedding_time = time.time() - embedding_start
            
            # 4. 사용자 인식
            recognition_start = time.time()
            recognition_result = self.recognize_user(face_embedding)
            recognition_time = time.time() - recognition_start
            
            total_time = time.time() - start_time
            
            # 결과 통합
            result = {
                'success': True,
                'faces_found': int(len(faces)),
                'main_face': {
                    **main_face,
                    **recognition_result
                },
                'all_faces': faces,
                'performance': {
                    'detection_time': float(round(detection_time, 3)),
                    'embedding_time': float(round(embedding_time, 3)),
                    'recognition_time': float(round(recognition_time, 3)),
                    'total_time': float(round(total_time, 3))
                },
                'method': 'yolov8_facenet',
                'embedding_size': len(face_embedding)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 이미지 처리 오류: {e}")
            return {
                'success': False,
                'faces_found': 0,
                'message': f'이미지 처리 중 오류: {str(e)}',
                'performance': {
                    'total_time': float(round(time.time() - start_time, 3))
                }
            }
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'system_type': 'yolov8_facenet',
            'detection_model': 'YOLOv8-Face',
            'recognition_model': 'FaceNet (512D)',
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'registered_users_count': len(self.user_names),
            'registered_users': self.user_names,
            'model_loaded': self.yolo_model is not None and self.facenet_model is not None,
            'yolo_loaded': self.yolo_model is not None,
            'facenet_loaded': self.facenet_model is not None,
            'embedding_dimension': 512
        }