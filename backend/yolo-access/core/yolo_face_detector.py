# YOLO + face_recognition 하이브리드 얼굴 인식 시스템

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import face_recognition
from pathlib import Path
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
import logging
from PIL import Image
import os
import pickle

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOFaceDetector:
    """YOLO (탐지) + face_recognition (인식) 하이브리드 시스템"""
    
    def __init__(self, model_type: str = "yolov8n"):
        """
        하이브리드 얼굴 인식기 초기화
        
        Args:
            model_type: YOLO 모델 타입
        """
        self.model_type = model_type
        self.yolo_model = None
        self.device = 'cpu'
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # YOLO 모델 로드
        self.load_yolo_model()
        
        # face_recognition용 데이터
        self.known_face_encodings = []
        self.known_face_names = []
        self.user_face_data = {}  # 각 사용자별 여러 임베딩 저장
        
        logger.info(f"✅ 하이브리드 얼굴 인식기 초기화 완료")
    
    def load_yolo_model(self):
        """YOLO 모델 로드 (얼굴 탐지용)"""
        try:
            logger.info(f"🔄 YOLO {self.model_type} 모델 로딩 중...")
            self.yolo_model = YOLO("yolov8n-face.pt")
            self.yolo_model.to(self.device)
            
            # 워밍업
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.yolo_model(dummy_image, verbose=False)
            
            logger.info(f"✅ YOLO 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ YOLO 모델 로드 실패: {e}")
            self.yolo_model = None
    
    def detect_faces_with_yolo(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        YOLO로 얼굴 탐지 (빠르고 정확한 위치 찾기)
        
        Returns:
            (얼굴 위치 리스트, 처리 시간)
        """
        if self.yolo_model is None:
            # YOLO 실패시 face_recognition으로 폴백
            return self.detect_faces_with_face_recognition(image)
        
        start_time = time.time()
        
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
                        class_name = self.yolo_model.names.get(class_id, "unknown")
                        
                        # 사람 클래스만 필터링
                        if class_name.lower() in ['person', '0']:
                            # YOLO는 전신을 탐지하므로 얼굴 영역 추정
                            # 상체 상단 1/4 영역을 얼굴로 추정
                            face_height = (y2 - y1) * 0.25
                            face_width = (x2 - x1) * 0.4
                            
                            # 얼굴 중심을 상체 상단으로 추정
                            center_x = (x1 + x2) / 2
                            face_y1 = y1
                            face_y2 = y1 + face_height
                            face_x1 = center_x - face_width / 2
                            face_x2 = center_x + face_width / 2
                            
                            # 이미지 경계 체크
                            h, w = image.shape[:2]
                            face_x1 = max(0, int(face_x1))
                            face_y1 = max(0, int(face_y1))
                            face_x2 = min(w, int(face_x2))
                            face_y2 = min(h, int(face_y2))
                            
                            if face_x2 > face_x1 and face_y2 > face_y1:
                                face_info = {
                                    'bbox': [face_x1, face_y1, face_x2-face_x1, face_y2-face_y1],
                                    'confidence': float(confidence),
                                    'method': 'yolo',
                                    'area': int((face_x2-face_x1) * (face_y2-face_y1))
                                }
                                faces.append(face_info)
            
            processing_time = time.time() - start_time
            
            # 면적 기준 정렬
            faces.sort(key=lambda x: x['area'], reverse=True)
            
            # YOLO로 얼굴을 못 찾으면 face_recognition으로 재시도
            if not faces:
                logger.info("🔄 YOLO 탐지 실패, face_recognition으로 재시도...")
                return self.detect_faces_with_face_recognition(image)
            
            return faces, float(processing_time)
            
        except Exception as e:
            logger.error(f"❌ YOLO 얼굴 탐지 오류: {e}")
            # 오류시 face_recognition으로 폴백
            return self.detect_faces_with_face_recognition(image)
    
    def detect_faces_with_face_recognition(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        face_recognition으로 얼굴 탐지 (YOLO 실패시 백업용)
        """
        start_time = time.time()
        
        try:
            # RGB 변환 (face_recognition은 RGB 사용)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 얼굴 위치 탐지
            face_locations = face_recognition.face_locations(rgb_image, model="hog")  # CNN은 느림
            
            faces = []
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_info = {
                    'bbox': [left, top, right-left, bottom-top],
                    'confidence': 0.9,  # face_recognition은 confidence 없음
                    'method': 'face_recognition',
                    'area': (right-left) * (bottom-top)
                }
                faces.append(face_info)
            
            processing_time = time.time() - start_time
            faces.sort(key=lambda x: x['area'], reverse=True)
            
            return faces, float(processing_time)
            
        except Exception as e:
            logger.error(f"❌ face_recognition 탐지 오류: {e}")
            return [], float(time.time() - start_time)
    
    def extract_face_encoding(self, image: np.ndarray, face_bbox: List[int]) -> np.ndarray:
        """
        face_recognition으로 128차원 얼굴 임베딩 추출 (고정밀)
        """
        try:
            x, y, w, h = face_bbox
            
            # 얼굴 영역 추출
            face_region = image[y:y+h, x:x+w]
            if face_region.size == 0:
                return np.array([])
            
            # RGB 변환
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # face_recognition으로 임베딩 추출
            encodings = face_recognition.face_encodings(rgb_face, model="large")  # 정확도 우선
            
            if encodings:
                return encodings[0]  # 128차원 벡터
            else:
                return np.array([])
                
        except Exception as e:
            logger.error(f"❌ 얼굴 인코딩 추출 오류: {e}")
            return np.array([])
    
    def load_user_database(self, users_dir: str = "data/users"):
        """사용자 데이터베이스 로드 (face_recognition 방식)"""
        try:
            if not os.path.exists(users_dir):
                logger.warning(f"사용자 데이터 폴더 없음: {users_dir}")
                return
            
            self.known_face_encodings = []
            self.known_face_names = []
            self.user_face_data = {}
            
            for user_name in os.listdir(users_dir):
                user_path = os.path.join(users_dir, user_name)
                
                if os.path.isdir(user_path):
                    logger.info(f"🔄 {user_name} 데이터 로드 중...")
                    
                    user_encodings = []
                    processed_count = 0
                    
                    for image_file in os.listdir(user_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(user_path, image_file)
                            
                            try:
                                # 이미지 로드
                                image = cv2.imread(image_path)
                                if image is None:
                                    continue
                                
                                # YOLO로 얼굴 탐지
                                faces, _ = self.detect_faces_with_yolo(image)
                                
                                if faces:
                                    # 가장 큰 얼굴 선택
                                    largest_face = max(faces, key=lambda x: x['area'])
                                    
                                    # face_recognition으로 인코딩 추출
                                    encoding = self.extract_face_encoding(image, largest_face['bbox'])
                                    
                                    if len(encoding) > 0:
                                        user_encodings.append(encoding)
                                        processed_count += 1
                                        
                            except Exception as e:
                                logger.warning(f"⚠️ {image_file} 처리 실패: {e}")
                                continue
                    
                    if user_encodings:
                        # 각 사진의 인코딩을 개별 저장 (평균내지 않음)
                        self.user_face_data[user_name] = user_encodings
                        
                        # 대표 인코딩 (첫 번째) 저장
                        self.known_face_encodings.append(user_encodings[0])
                        self.known_face_names.append(user_name)
                        
                        logger.info(f"✅ {user_name}: {processed_count}장 처리 완료")
                    else:
                        logger.warning(f"⚠️ {user_name}: 처리 가능한 이미지 없음")
            
            logger.info(f"🎯 총 {len(self.known_face_names)}명의 사용자 데이터 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ 사용자 데이터베이스 로드 실패: {e}")
    
    def recognize_user(self, face_encoding: np.ndarray, tolerance: float = 0.6) -> Dict:
        """
        face_recognition으로 사용자 인식 (고정밀)
        """
        if len(self.known_face_encodings) == 0:
            return {
                'recognized': False,
                'user_name': '데이터베이스 없음',
                'confidence': 0.0,
                'message': '등록된 사용자 데이터가 없습니다'
            }
        
        if len(face_encoding) == 0:
            return {
                'recognized': False,
                'user_name': '특징 추출 실패',
                'confidence': 0.0,
                'message': '얼굴 특징을 추출할 수 없습니다'
            }
        
        try:
            # face_recognition의 compare_faces 사용
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=tolerance
            )
            
            # 거리 계산
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]
                confidence = round((1 - best_distance) * 100, 2)
                
                # 매칭 성공시
                if matches[best_match_index] and best_distance < tolerance:
                    user_name = self.known_face_names[best_match_index]
                    
                    # 해당 사용자의 모든 사진과 추가 비교 (정확도 향상)
                    if user_name in self.user_face_data:
                        user_distances = face_recognition.face_distance(
                            self.user_face_data[user_name], 
                            face_encoding
                        )
                        avg_distance = np.mean(user_distances)
                        final_confidence = round((1 - avg_distance) * 100, 2)
                    else:
                        final_confidence = confidence
                    
                    return {
                        'recognized': True,
                        'user_name': str(user_name),
                        'confidence': float(final_confidence),
                        'distance': float(best_distance),
                        'tolerance': float(tolerance),
                        'message': f'{user_name} 인식 성공'
                    }
                else:
                    return {
                        'recognized': False,
                        'user_name': '미등록 인물',
                        'confidence': float(confidence),
                        'distance': float(best_distance),
                        'tolerance': float(tolerance),
                        'best_match': str(self.known_face_names[best_match_index]),
                        'message': f'거리 {round(best_distance, 3)} > 허용값 {tolerance}'
                    }
            else:
                return {
                    'recognized': False,
                    'user_name': '비교 실패',
                    'confidence': 0.0,
                    'message': '거리 계산 실패'
                }
                
        except Exception as e:
            logger.error(f"❌ 사용자 인식 오류: {e}")
            return {
                'recognized': False,
                'user_name': '인식 오류',
                'confidence': 0.0,
                'message': f'인식 처리 중 오류: {str(e)}'
            }
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        하이브리드 이미지 처리 파이프라인
        1. YOLO로 빠른 얼굴 탐지
        2. face_recognition으로 정확한 인식
        """
        start_time = time.time()
        
        try:
            # 1. YOLO로 얼굴 탐지
            faces, detection_time = self.detect_faces_with_yolo(image)
            
            if not faces:
                return {
                    'success': False,
                    'faces_found': 0,
                    'message': '얼굴을 찾을 수 없습니다',
                    'detection_time': float(round(detection_time, 3)),
                    'total_time': float(round(time.time() - start_time, 3))
                }
            
            # 2. 가장 큰 얼굴 선택
            main_face = faces[0]
            
            # 3. face_recognition으로 인코딩 추출
            encoding_start = time.time()
            face_encoding = self.extract_face_encoding(image, main_face['bbox'])
            encoding_time = time.time() - encoding_start
            
            # 4. 사용자 인식
            recognition_start = time.time()
            recognition_result = self.recognize_user(face_encoding)
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
                    'encoding_time': float(round(encoding_time, 3)),
                    'recognition_time': float(round(recognition_time, 3)),
                    'total_time': float(round(total_time, 3))
                },
                'method': 'hybrid_yolo_face_recognition'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 이미지 처리 오류: {e}")
            return {
                'success': False,
                'faces_found': 0,
                'message': f'이미지 처리 중 오류: {str(e)}',
                'total_time': float(round(time.time() - start_time, 3))
            }
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환 (기존 main.py와 호환)"""
        return {
            # 기존 키들 (main.py 호환)
            'model_type': self.model_type,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'registered_users_count': len(self.known_face_names),
            'registered_users': self.known_face_names,
            'model_loaded': self.yolo_model is not None,
            
            # 새로운 하이브리드 정보
            'detection_model': self.model_type,
            'recognition_model': 'face_recognition (dlib)',
            'face_recognition_tolerance': 0.6,
            'yolo_loaded': self.yolo_model is not None,
            'face_recognition_loaded': len(self.known_face_encodings) > 0,
            'system_type': 'hybrid_yolo_face_recognition'
        }