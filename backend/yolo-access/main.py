# YOLO 기반 출입 관리 FastAPI 서버

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import asyncio

# YOLO 얼굴 탐지기 임포트
from core.yolo_face_detector import YOLOFaceDetector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="YOLO 기반 출입 관리 API",
    description="YOLOv5-Face/YOLOv8을 사용한 스마트 출입 관리 시스템",
    version="2.0.0"
)

# CORS 설정 (모바일 앱 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영시에는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델들
class AccessLogRequest(BaseModel):
    user_name: str
    confidence: float
    decision: str  # 'approved', 'denied', 'manual_approved'
    approved_by: str  # 'system', 'staff'
    reason: Optional[str] = None
    timestamp: str
    staff_id: Optional[str] = None
    processing_time: Optional[float] = None

class SystemSettings(BaseModel):
    model_type: str = "yolov8n"
    confidence_threshold: float = 0.5
    recognition_threshold: float = 0.7

# 전역 변수들
face_detector: Optional[YOLOFaceDetector] = None
access_logs: List[Dict] = []
system_settings = SystemSettings()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global face_detector
    
    try:
        # 로그 디렉토리 생성
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/users", exist_ok=True)  # users 폴더로 변경
        
        logger.info("🚀 YOLO 기반 출입 관리 시스템 시작")
        
        # YOLO 얼굴 탐지기 초기화
        logger.info(f"🔄 YOLO 모델 초기화 중... (모델: {system_settings.model_type})")
        face_detector = YOLOFaceDetector(model_type=system_settings.model_type)
        
        # 사용자 데이터베이스 로드
        logger.info("📚 사용자 데이터베이스 로드 중...")
        face_detector.load_user_database()
        
        # 기존 로그 파일 로드
        load_existing_logs()
        
        logger.info("✅ 서버 초기화 완료")
        
    except Exception as e:
        logger.error(f"❌ 서버 초기화 실패: {e}")
        raise

def load_existing_logs():
    """기존 출입 기록 로드"""
    global access_logs
    
    try:
        log_file = "logs/access_logs.json"
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                access_logs = json.load(f)
            logger.info(f"📋 기존 출입 기록 {len(access_logs)}건 로드")
        else:
            access_logs = []
            logger.info("📋 새로운 출입 기록 시작")
            
    except Exception as e:
        logger.error(f"❌ 출입 기록 로드 실패: {e}")
        access_logs = []

def save_logs_to_file():
    """출입 기록을 파일에 저장"""
    try:
        with open("logs/access_logs.json", 'w', encoding='utf-8') as f:
            json.dump(access_logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"❌ 로그 저장 실패: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지"""
    if face_detector is None:
        return HTMLResponse("""
        <html>
            <body>
                <h1>❌ YOLO 모델 로딩 실패</h1>
                <p>서버를 다시 시작해주세요.</p>
            </body>
        </html>
        """)
    
    model_info = face_detector.get_model_info()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO 기반 스마트 출입 관리</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .status {{ background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .info-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .api-section {{ margin-top: 30px; }}
            .endpoint {{ background: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏢 YOLO 기반 스마트 출입 관리</h1>
                <p>YOLOv5-Face/YOLOv8을 활용한 AI 얼굴 인식 출입 관리 시스템</p>
            </div>
            
            <div class="status">
                <h3>✅ 시스템 상태</h3>
                <p><strong>탐지 모델:</strong> {model_info['detection_model']}</p>
                <p><strong>인식 모델:</strong> {model_info['recognition_model']}</p>
                <p><strong>처리 장치:</strong> {model_info['device']}</p>
                <p><strong>등록된 사용자:</strong> {model_info['registered_users_count']}명</p>
                <p><strong>총 출입 기록:</strong> {len(access_logs)}건</p>
            </div>
            
            <div class="info-grid">
                <div class="info-card">
                    <h4>👥 등록된 사용자</h4>
                    {'<br>'.join(['• ' + name for name in model_info['registered_users']]) if model_info['registered_users'] else '사용자 데이터를 추가해주세요'}
                </div>
                
                <div class="info-card">
                    <h4>⚙️ AI 모델 설정</h4>
                    <p><strong>얼굴 탐지 임계값:</strong> {model_info['confidence_threshold']}</p>
                    <p><strong>IOU 임계값:</strong> {model_info['iou_threshold']}</p>
                    <p><strong>인식 임계값:</strong> 70%</p>
                </div>
            </div>
            
            <div class="api-section">
                <h3>🔗 API 엔드포인트</h3>
                <div class="endpoint">
                    <strong>POST /recognize</strong> - 얼굴 인식 (이미지 업로드)
                </div>
                <div class="endpoint">
                    <strong>POST /log_access</strong> - 출입 기록 저장
                </div>
                <div class="endpoint">
                    <strong>GET /logs</strong> - 출입 기록 조회
                </div>
                <div class="endpoint">
                    <strong>GET /users</strong> - 등록된 사용자 목록 조회
                </div>
                <div class="endpoint">
                    <strong>GET /docs</strong> - API 문서 (Swagger UI)
                </div>
            </div>
            
            <div class="api-section">
                <h3>📱 사용 방법</h3>
                <ol>
                    <li>모바일 앱에서 카메라로 얼굴 촬영</li>
                    <li>POST /recognize 엔드포인트로 이미지 전송</li>
                    <li>AI가 얼굴 탐지 및 사용자 인식</li>
                    <li>시스템에서 자동 승인/거부 또는 수동 결정</li>
                    <li>POST /log_access로 출입 기록 저장</li>
                </ol>
            </div>
            
            <div class="api-section">
                <h3>🎯 시스템 특징</h3>
                <ul>
                    <li>✅ <strong>실시간 AI 얼굴 인식:</strong> YOLOv8 기반 고속 처리</li>
                    <li>🔒 <strong>보안성:</strong> 등록된 사용자만 출입 허용</li>
                    <li>📊 <strong>출입 통계:</strong> 실시간 출입 현황 및 분석</li>
                    <li>📱 <strong>모바일 연동:</strong> 스마트폰 앱으로 간편 관리</li>
                    <li>⚡ <strong>빠른 처리:</strong> CPU 환경에서도 1-3초 내 인식</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(html_content)

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    YOLO 기반 얼굴 인식
    
    모바일 앱에서 촬영한 사진을 받아서 등록된 사용자 인식 수행
    """
    if face_detector is None:
        raise HTTPException(status_code=503, detail="YOLO 모델이 로드되지 않았습니다")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    try:
        # 이미지 읽기 및 전처리
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # OpenCV 형식으로 변환
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 이미지 크기 최적화 (성능 향상)
        height, width = image_np.shape[:2]
        original_size = f"{width}x{height}"
        
        if width > 1024:
            scale = 1024 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_np = cv2.resize(image_np, (new_width, new_height))
            resized_size = f"{new_width}x{new_height}"
        else:
            resized_size = original_size
        
        logger.info(f"📸 이미지 처리 시작: {original_size} -> {resized_size}")
        
        # YOLO 기반 얼굴 인식 수행
        result = face_detector.process_image(image_np)
        
        # 결과 로깅
        if result['success']:
            main_face = result['main_face']
            logger.info(f"✅ 인식 완료: {main_face['user_name']} (신뢰도: {main_face.get('confidence', 0)}%)")
        else:
            logger.warning(f"⚠️ 인식 실패: {result['message']}")
        
        # 이미지 정보 추가
        result['image_info'] = {
            'original_size': original_size,
            'processed_size': resized_size,
            'file_size': len(image_data),
            'format': image.format
        }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 얼굴 인식 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/log_access")
async def log_access(log_request: AccessLogRequest, background_tasks: BackgroundTasks):
    """출입 기록 저장"""
    try:
        # 출입 로그 생성
        access_log = {
            "id": len(access_logs) + 1,
            "timestamp": log_request.timestamp,
            "user_name": log_request.user_name,
            "confidence": log_request.confidence,
            "decision": log_request.decision,
            "approved_by": log_request.approved_by,
            "reason": log_request.reason,
            "staff_id": log_request.staff_id,
            "processing_time": log_request.processing_time,
            "created_at": datetime.now().isoformat()
        }
        
        # 메모리에 추가
        access_logs.append(access_log)
        
        # 백그라운드에서 파일 저장
        background_tasks.add_task(save_logs_to_file)
        
        logger.info(f"📝 출입 기록 저장: {log_request.user_name} - {log_request.decision}")
        
        return {
            "success": True,
            "message": "출입 기록이 저장되었습니다",
            "log_id": access_log["id"]
        }
        
    except Exception as e:
        logger.error(f"❌ 출입 기록 저장 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"출입 기록 저장 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/logs")
async def get_logs(limit: int = 50, offset: int = 0):
    """출입 기록 조회"""
    try:
        total_count = len(access_logs)
        
        # 최신순 정렬
        sorted_logs = sorted(access_logs, key=lambda x: x["timestamp"], reverse=True)
        
        # 페이지네이션
        paginated_logs = sorted_logs[offset:offset + limit]
        
        # 통계 계산
        approved_count = sum(1 for log in access_logs if log["decision"] in ["approved", "manual_approved"])
        denied_count = sum(1 for log in access_logs if log["decision"] == "denied")
        
        return {
            "logs": paginated_logs,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count
            },
            "statistics": {
                "total_attempts": total_count,
                "approved": approved_count,
                "denied": denied_count,
                "approval_rate": round((approved_count / total_count * 100), 2) if total_count > 0 else 0,
                "today_count": len([log for log in access_logs if log["timestamp"].startswith(datetime.now().strftime("%Y-%m-%d"))])
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 출입 기록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"출입 기록 조회 중 오류: {str(e)}")

@app.get("/users")
async def get_users():
    """등록된 사용자 목록 조회"""
    if face_detector is None:
        raise HTTPException(status_code=503, detail="YOLO 모델이 로드되지 않았습니다")
    
    try:
        model_info = face_detector.get_model_info()
        
        users = []
        for name in model_info['registered_users']:
            users.append({
                "name": name,
                "department": "직원",  # 실제로는 DB에서 가져와야 함
                "status": "active",
                "registered_date": "2024-01-01"  # 실제로는 등록 날짜
            })
        
        return {
            "users": users,
            "total_count": len(users),
            "model_info": {
                "model_type": model_info['model_type'],
                "device": model_info['device'],
                "confidence_threshold": model_info['confidence_threshold']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 사용자 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"사용자 목록 조회 중 오류: {str(e)}")

@app.get("/system/status")
async def get_system_status():
    """시스템 상태 조회"""
    if face_detector is None:
        return {
            "status": "error",
            "message": "YOLO 모델이 로드되지 않았습니다",
            "model_loaded": False
        }
    
    model_info = face_detector.get_model_info()
    
    return {
        "status": "running",
        "server_time": datetime.now().isoformat(),
        "model_info": model_info,
        "statistics": {
            "total_logs": len(access_logs),
            "registered_users": len(model_info['registered_users']),
            "system_type": "face_recognition_access_control"
        }
    }

@app.post("/system/reload_users")
async def reload_users():
    """사용자 데이터베이스 다시 로드"""
    if face_detector is None:
        raise HTTPException(status_code=503, detail="YOLO 모델이 로드되지 않았습니다")
    
    try:
        logger.info("🔄 사용자 데이터베이스 다시 로드 중...")
        face_detector.load_user_database()
        
        model_info = face_detector.get_model_info()
        
        logger.info(f"✅ 사용자 데이터베이스 다시 로드 완료: {model_info['registered_users_count']}명")
        
        return {
            "success": True,
            "message": "사용자 데이터베이스가 다시 로드되었습니다",
            "users_loaded": model_info['registered_users_count'],
            "user_names": model_info['registered_users']
        }
        
    except Exception as e:
        logger.error(f"❌ 사용자 데이터베이스 다시 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"다시 로드 중 오류: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": face_detector is not None,
        "version": "2.0.0",
        "system_type": "access_control"
    }

@app.get("/test")
async def test_endpoint():
    """테스트용 엔드포인트"""
    if face_detector is None:
        return {"error": "YOLO 모델이 로드되지 않았습니다"}
    
    # 더미 이미지로 테스트
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        result = face_detector.process_image(dummy_image)
        return {
            "test_status": "success",
            "message": "YOLO 모델이 정상적으로 작동합니다",
            "dummy_result": result
        }
    except Exception as e:
        return {
            "test_status": "error",
            "message": f"YOLO 모델 테스트 실패: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🏢 YOLO 기반 스마트 출입 관리 시스템")
    print("="*60)
    print("📋 주요 기능:")
    print("  • YOLOv5-Face/YOLOv8 기반 고정밀 얼굴 탐지")
    print("  • 등록된 직원/회원 자동 인식")
    print("  • 실시간 출입 승인/거부 시스템")
    print("  • 스마트폰 앱과 실시간 연동")
    print("  • 자동 출입 기록 및 통계")
    print("\n🔧 기술 스택:")
    print("  • FastAPI + YOLO + PyTorch")
    print("  • OpenCV + NumPy")
    print("  • CPU 최적화")
    print("\n🚀 서버 시작 중...")
    print("📍 웹 인터페이스: http://localhost:8001")
    print("📍 API 문서: http://localhost:8001/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        log_level="info",
        access_log=True
    )