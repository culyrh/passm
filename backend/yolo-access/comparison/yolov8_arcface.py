# YOLOv8-Face + ArcFace 얼굴 인식 시스템

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
import dlib
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

class ArcFaceModel(nn.Module):
    """ArcFace 모델 (간단한 ResNet50 기반 구현)"""
    
    def __init__(self, embedding_size=512, num_classes=None):
        super(ArcFaceModel, self).__init__()
        self.embedding_size = embedding_size
        
        # ResNet50 기반 백본 (간단화)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet 블록들 (간단화)
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_size)
        
        # ArcFace 헤드 (훈련용, 추론시에는 사용 안함)
        if num_classes:
            self.arc_head = ArcMarginProduct(embedding_size, num_classes)
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # L2 정규화 (중요!)
        x = F.normalize(x, p=2, dim=1)
        
        return x

class ArcMarginProduct(nn.Module):
    """ArcFace 손실을 위한 마진 곱셈 레이어"""
    
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if label is None:
            return cosine * self.s
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * np.cos(self.m) - sine * np.sin(self.m)
        
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

class YOLOv8ArcFace:
    """YOLOv8-Face + ArcFace 하이브리드 시스템"""
    
    def __init__(self, 
                 yolo_model_path: str = "yolov8n-face.pt",
                 arcface_model_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Args:
            yolo_model_path: YOLOv8-Face 모델 경로
            arcface_model_path: ArcFace 모델 경로 (None이면 기본 모델)
            device: 연산 장치
        """
        self.device = device
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # 모델 로드
        self.load_yolo_model(yolo_model_path)
        self.load_arcface_model(arcface_model_path)
        
        # 얼굴 정렬용 dlib (ArcFace의 핵심 특징)
        self.load_face_alignment()
        
        # 사용자 데이터
        self.user_embeddings = {}
        self.user_names = []
        
        # 이미지 전처리 (ArcFace 표준)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),  # ArcFace 표준 크기
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info("✅ YOLOv8-Face + ArcFace 시스템 초기화 완료")
    
    def load_yolo_model(self, model_path: str):
        """YOLOv8-Face 모델 로드"""
        try:
            logger.info(f"📁 YOLOv8-Face 모델 로드 중: {model_path}")
            
            if not os.path.exists(model_path):
                logger.warning(f"모델 파일 없음. 기본 YOLOv8n 다운로드")
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
    
    def load_arcface_model(self, model_path: Optional[str]):
        """ArcFace 모델 로드"""
        try:
            logger.info("📁 ArcFace 모델 로드 중...")
            
            self.arcface_model = ArcFaceModel(embedding_size=512)
            self.arcface_model.to(self.device)
            self.arcface_model.eval()
            
            # 사전 훈련된 가중치가 있다면 로드
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.arcface_model.load_state_dict(checkpoint)
                logger.info(f"✅ 사전 훈련된 ArcFace 가중치 로드: {model_path}")
            else:
                logger.warning("⚠️ 사전 훈련된 가중치 없음. 랜덤 초기화된 모델 사용")
            
            logger.info("✅ ArcFace 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ ArcFace 모델 로드 실패: {e}")
            self.arcface_model = None
    
    def load_face_alignment(self):
        """얼굴 정렬용 dlib 로드 (ArcFace의 핵심 기능)"""
        try:
            logger.info("📁 얼굴 정렬 모델 로드 중...")
        
            # dlib 얼굴 검출기
            self.face_detector = dlib.get_frontal_face_detector()
        
            # 현재 파일의 디렉토리 기준으로 경로 설정
            import os
            current_dir = os.path.dirname(__file__)
            predictor_path = os.path.join(current_dir, "models", "shape_predictor_68_face_landmarks.dat")
        
            # 경로 확인 로그 추가
            logger.info(f"🔍 랜드마크 모델 경로: {predictor_path}")
            logger.info(f"🔍 파일 존재 여부: {os.path.exists(predictor_path)}")
        
            if os.path.exists(predictor_path):
                self.shape_predictor = dlib.shape_predictor(predictor_path)
                logger.info("✅ 68점 랜드마크 예측기 로드")
            else:
                # 5점 모델도 시도
                predictor_path_5 = os.path.join(current_dir, "models", "shape_predictor_5_face_landmarks.dat")
                if os.path.exists(predictor_path_5):
                    self.shape_predictor = dlib.shape_predictor(predictor_path_5)
                    logger.info("✅ 5점 랜드마크 예측기 로드")
                else:
                    logger.warning("⚠️ dlib 랜드마크 예측기 없음. 정렬 기능 비활성화")
                    self.shape_predictor = None
                
        except Exception as e:
            logger.error(f"❌ 얼굴 정렬 모델 로드 실패: {e}")
            self.face_detector = None
            self.shape_predictor = None
    
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
    
    def align_face(self, image: np.ndarray, face_bbox: List[int]) -> np.ndarray:
        """
        얼굴 정렬 (ArcFace의 핵심 기능)
        - 얼굴을 정면으로 회전
        - 눈, 코, 입의 위치를 표준화
        """
        try:
            x, y, w, h = face_bbox
            face_region = image[y:y+h, x:x+w]
            
            if self.face_detector is None or self.shape_predictor is None:
                # 정렬 불가능하면 리사이즈만
                return cv2.resize(face_region, (112, 112))
            
            # dlib으로 랜드마크 검출
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            if len(faces) == 0:
                return cv2.resize(face_region, (112, 112))
            
            # 첫 번째 얼굴의 랜드마크
            landmarks = self.shape_predictor(gray, faces[0])
            
            # 랜드마크를 numpy 배열로 변환
            points = []
            for n in range(landmarks.num_parts):
                points.append([landmarks.part(n).x, landmarks.part(n).y])
            points = np.array(points)
            
            # 눈 중심점 계산 (정렬 기준점)
            if len(points) >= 68:  # 68점 모델
                left_eye = points[36:42].mean(axis=0)
                right_eye = points[42:48].mean(axis=0)
            elif len(points) >= 5:  # 5점 모델
                left_eye = points[0]
                right_eye = points[1]
            else:
                return cv2.resize(face_region, (112, 112))
            
            # 회전 각도 계산
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # 두 눈 사이의 거리
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # 얼굴 중심점
            eye_center = ((left_eye[0] + right_eye[0]) // 2, 
                         (left_eye[1] + right_eye[1]) // 2)
            
            # 회전 및 크기 조정을 위한 변환 행렬
            desired_eye_distance = 35  # 목표 눈 간격 (픽셀)
            scale = desired_eye_distance / eye_distance
            
            # 회전 변환
            M = cv2.getRotationMatrix2D(eye_center, angle, scale)
            
            # 이동 변환 (얼굴을 중앙으로)
            tX = 112 * 0.5
            tY = 112 * 0.35  # 약간 위쪽에 위치
            M[0, 2] += (tX - eye_center[0])
            M[1, 2] += (tY - eye_center[1])
            
            # 변환 적용
            aligned_face = cv2.warpAffine(face_region, M, (112, 112), 
                                        flags=cv2.INTER_CUBIC)
            
            return aligned_face
            
        except Exception as e:
            logger.error(f"❌ 얼굴 정렬 오류: {e}")
            # 정렬 실패시 단순 리사이즈
            try:
                x, y, w, h = face_bbox
                face_region = image[y:y+h, x:x+w]
                return cv2.resize(face_region, (112, 112))
            except:
                return np.zeros((112, 112, 3), dtype=np.uint8)
    
    def extract_face_embedding(self, image: np.ndarray, face_bbox: List[int]) -> np.ndarray:
        """ArcFace로 얼굴 임베딩 추출 (정렬 포함)"""
        if self.arcface_model is None:
            return np.array([])
        
        try:
            # 1. 얼굴 정렬 (ArcFace의 핵심!)
            aligned_face = self.align_face(image, face_bbox)
            
            # 2. BGR -> RGB 변환
            face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # 3. 전처리
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # 4. ArcFace 임베딩 추출
            with torch.no_grad():
                embedding = self.arcface_model(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ ArcFace 임베딩 추출 오류: {e}")
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
                                    
                                    # ArcFace 임베딩 추출 (정렬 포함)
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
    
    def save_embeddings(self, save_path: str = "data/arcface_embeddings.pkl"):
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
    
    def load_embeddings(self, load_path: str = "data/arcface_embeddings.pkl"):
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
    
    def recognize_user(self, face_embedding: np.ndarray, threshold: float = 0.4) -> Dict:
        """
        ArcFace 임베딩으로 사용자 인식 - 수정된 버전
        """
        if len(self.user_embeddings) == 0:
            return {
                'recognized': False,
                'user_name': '데이터베이스 없음',
                'confidence': 0.0,
                'similarity': 0.0,
                'message': '등록된 사용자 데이터가 없습니다'
            }
    
        if len(face_embedding) == 0:
            return {
                'recognized': False,
                'user_name': '특징 추출 실패',
                'confidence': 0.0,
                'similarity': 0.0,
                'message': '얼굴 특징을 추출할 수 없습니다'
            }
    
        try:
            # 임베딩 정규화
            face_embedding = face_embedding.astype(np.float64)
            face_norm = np.linalg.norm(face_embedding)
            if face_norm > 0:
                face_embedding = face_embedding / face_norm
        
            best_distance = float('inf')
            best_user = None
            all_similarities = {}
        
            # 각 사용자와의 거리 계산
            for user_name, embeddings_list in self.user_embeddings.items():
                distances = []
            
                for stored_embedding in embeddings_list:
                    # 저장된 임베딩 정규화
                    stored_embedding = stored_embedding.astype(np.float64)
                    stored_norm = np.linalg.norm(stored_embedding)
                    if stored_norm > 0:
                        stored_embedding = stored_embedding / stored_norm
                
                    # 유클리디안 거리 계산 (코사인 유사도 대신)
                    distance = np.linalg.norm(face_embedding - stored_embedding)
                    distances.append(distance)
            
                if distances:
                    min_distance = min(distances)
                    # 거리를 유사도로 변환 (0~1 범위)
                    similarity = max(0, 1 - min_distance / 2.0)
                    all_similarities[user_name] = similarity
                
                    if min_distance < best_distance:
                        best_distance = min_distance
                        best_user = user_name
        
            # 최종 유사도 및 신뢰도
            best_similarity = max(0, 1 - best_distance / 2.0)
            confidence = best_similarity * 100
        
            # 인식 성공 조건: 유사도가 threshold보다 높고, 거리가 1.2보다 작을 때
            if best_similarity > threshold and best_distance < 1.2:
                return {
                    'recognized': True,
                    'user_name': str(best_user),
                    'confidence': float(round(confidence, 2)),
                    'similarity': float(round(best_similarity, 4)),
                    'distance': float(round(best_distance, 4)),
                    'threshold': float(threshold),
                    'all_similarities': {k: float(round(v, 4)) for k, v in all_similarities.items()},
                    'message': f'{best_user} 인식 성공 (거리 기반)',
                    'method': 'euclidean_distance'
                }
            else:
                return {
                    'recognized': False,
                    'user_name': '미등록 인물',
                    'confidence': float(round(confidence, 2)),
                    'similarity': float(round(best_similarity, 4)),
                    'distance': float(round(best_distance, 4)),
                    'threshold': float(threshold),
                    'best_match': str(best_user) if best_user else '매칭 없음',
                    'all_similarities': {k: float(round(v, 4)) for k, v in all_similarities.items()},
                    'message': f'거리 {round(best_distance, 4)} > 임계값, 유사도 {round(best_similarity, 4)} < {threshold}',
                    'method': 'euclidean_distance'
                }
            
        except Exception as e:
            logger.error(f"❌ 사용자 인식 오류: {e}")
            return {
                'recognized': False,
                'user_name': '인식 오류',
                'confidence': 0.0,
                'similarity': 0.0,
                'message': f'인식 처리 중 오류: {str(e)}'
            }
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        전체 이미지 처리 파이프라인
        1. YOLOv8-Face로 얼굴 검출
        2. 얼굴 정렬 (ArcFace 핵심 기능)
        3. ArcFace로 임베딩 추출 및 인식
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
            
            # 3. 얼굴 정렬 시간 측정
            alignment_start = time.time()
            aligned_face = self.align_face(image, main_face['bbox'])
            alignment_time = time.time() - alignment_start
            
            # 4. ArcFace 임베딩 추출
            embedding_start = time.time()
            face_embedding = self.extract_face_embedding(image, main_face['bbox'])
            embedding_time = time.time() - embedding_start
            
            # 5. 사용자 인식
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
                    **recognition_result,
                    'aligned': True  # 정렬되었음을 표시
                },
                'all_faces': faces,
                'performance': {
                    'detection_time': float(round(detection_time, 3)),
                    'alignment_time': float(round(alignment_time, 3)),
                    'embedding_time': float(round(embedding_time, 3)),
                    'recognition_time': float(round(recognition_time, 3)),
                    'total_time': float(round(total_time, 3))
                },
                'method': 'yolov8_arcface',
                'embedding_size': len(face_embedding),
                'features': {
                    'face_alignment': True,
                    'landmark_detection': self.shape_predictor is not None,
                    'cosine_similarity': True
                }
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
            'system_type': 'yolov8_arcface',
            'detection_model': 'YOLOv8-Face',
            'recognition_model': 'ArcFace (512D)',
            'alignment_model': 'dlib landmarks',
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'registered_users_count': len(self.user_names),
            'registered_users': self.user_names,
            'model_loaded': self.yolo_model is not None and self.arcface_model is not None,
            'yolo_loaded': self.yolo_model is not None,
            'arcface_loaded': self.arcface_model is not None,
            'alignment_loaded': self.shape_predictor is not None,
            'embedding_dimension': 512,
            'similarity_metric': 'cosine',
            'features': {
                'face_alignment': True,
                'landmark_standardization': True,
                'rotation_correction': True,
                'reference_point_unification': True
            }
        }