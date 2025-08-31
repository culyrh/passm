# YOLOv8 + FaceNet vs YOLOv8 + ArcFace 성능 비교 서버

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import io
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
import logging
import asyncio
import concurrent.futures

# 우리가 만든 모델들 import
from yolov8_facenet import YOLOv8FaceNet
from yolov8_arcface import YOLOv8ArcFace

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comparison_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="YOLOv8 + FaceNet vs ArcFace 성능 비교",
    description="논문 기반 얼굴 인식 시스템 성능 비교 분석",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델들
class ComparisonRequest(BaseModel):
    test_name: str
    description: Optional[str] = None
    models_to_test: List[Literal["facenet", "arcface"]] = ["facenet", "arcface"]

class BenchmarkResult(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_processing_time: float
    avg_detection_time: float
    avg_recognition_time: float
    total_tests: int
    successful_recognitions: int
    failed_recognitions: int

# 전역 변수들
facenet_model: Optional[YOLOv8FaceNet] = None
arcface_model: Optional[YOLOv8ArcFace] = None
comparison_results: List[Dict] = []
benchmark_history: List[Dict] = []

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 두 모델 모두 로드"""
    global facenet_model, arcface_model
    
    try:
        logger.info("🚀 얼굴 인식 성능 비교 서버 시작")
        
        # 로그 디렉토리 생성
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/users", exist_ok=True)
        os.makedirs("data/comparison_results", exist_ok=True)
        
        # 두 모델 병렬 로드
        logger.info("📦 FaceNet과 ArcFace 모델 병렬 로드 시작...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            facenet_future = executor.submit(load_facenet_model)
            arcface_future = executor.submit(load_arcface_model)
            
            facenet_model = facenet_future.result()
            arcface_model = arcface_future.result()
        
        # 사용자 데이터베이스 로드
        if facenet_model:
            logger.info("📚 FaceNet 사용자 데이터베이스 로드...")
            if not facenet_model.load_embeddings():
                facenet_model.load_user_database()
        
        if arcface_model:
            logger.info("📚 ArcFace 사용자 데이터베이스 로드...")
            if not arcface_model.load_embeddings():
                arcface_model.load_user_database()
        
        # 기존 비교 결과 로드
        load_existing_results()
        
        logger.info("✅ 서버 초기화 완료!")
        logger.info(f"   - FaceNet 로드: {'✅' if facenet_model else '❌'}")
        logger.info(f"   - ArcFace 로드: {'✅' if arcface_model else '❌'}")
        
    except Exception as e:
        logger.error(f"❌ 서버 초기화 실패: {e}")
        raise

def load_facenet_model():
    """FaceNet 모델 로드"""
    try:
        model = YOLOv8FaceNet(
            yolo_model_path="yolov8n-face.pt",
            device='cpu'
        )
        logger.info("✅ FaceNet 모델 로드 완료")
        return model
    except Exception as e:
        logger.error(f"❌ FaceNet 모델 로드 실패: {e}")
        return None

def load_arcface_model():
    """ArcFace 모델 로드"""
    try:
        model = YOLOv8ArcFace(
            yolo_model_path="yolov8n-face.pt",
            device='cpu'
        )
        logger.info("✅ ArcFace 모델 로드 완료")
        return model
    except Exception as e:
        logger.error(f"❌ ArcFace 모델 로드 실패: {e}")
        return None

def load_existing_results():
    """기존 비교 결과 로드"""
    global comparison_results, benchmark_history
    
    try:
        results_file = "data/comparison_results/results.json"
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                comparison_results = data.get('comparison_results', [])
                benchmark_history = data.get('benchmark_history', [])
            logger.info(f"📂 기존 비교 결과 {len(comparison_results)}건 로드")
        else:
            comparison_results = []
            benchmark_history = []
            logger.info("📂 새로운 비교 결과 시작")
    except Exception as e:
        logger.error(f"❌ 결과 로드 실패: {e}")
        comparison_results = []
        benchmark_history = []

def save_results():
    """비교 결과 저장"""
    try:
        with open("data/comparison_results/results.json", 'w', encoding='utf-8') as f:
            json.dump({
                'comparison_results': comparison_results,
                'benchmark_history': benchmark_history,
                'last_updated': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"❌ 결과 저장 실패: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지"""
    facenet_status = "✅ 로드됨" if facenet_model else "❌ 로드 실패"
    arcface_status = "✅ 로드됨" if arcface_model else "❌ 로드 실패"
    
    facenet_users = len(facenet_model.user_names) if facenet_model else 0
    arcface_users = len(arcface_model.user_names) if arcface_model else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8 + FaceNet vs ArcFace 성능 비교</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; backdrop-filter: blur(10px); }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .models-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0; }}
            .model-card {{ background: rgba(255,255,255,0.15); padding: 25px; border-radius: 10px; }}
            .comparison-section {{ background: rgba(255,255,255,0.1); padding: 25px; border-radius: 10px; margin: 20px 0; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .stat-item {{ background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center; }}
            .api-section {{ margin-top: 30px; }}
            .endpoint {{ background: rgba(255,255,255,0.1); padding: 15px; margin: 10px 0; border-radius: 8px; }}
            .highlight {{ color: #ffd700; font-weight: bold; }}
            .vs {{ font-size: 2em; text-align: center; margin: 20px 0; color: #ffd700; }}
            .feature-list {{ list-style: none; padding: 0; }}
            .feature-list li {{ margin: 8px 0; padding-left: 20px; position: relative; }}
            .feature-list li:before {{ content: "✓"; position: absolute; left: 0; color: #4ade80; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 YOLOv8 + FaceNet vs ArcFace</h1>
                <p>논문 "Personal Verification System Using ID Card and Face Photo" 기반 성능 비교</p>
                <p class="highlight">두 모델의 정확도와 처리 속도를 실시간으로 비교 분석합니다</p>
            </div>
            
            <div class="models-grid">
                <div class="model-card">
                    <h3>🤖 YOLOv8 + FaceNet</h3>
                    <p><strong>상태:</strong> {facenet_status}</p>
                    <p><strong>등록 사용자:</strong> {facenet_users}명</p>
                    <p><strong>임베딩 차원:</strong> 512차원</p>
                    <p><strong>유사도 측정:</strong> 유클리드 거리</p>
                    
                    <h4>📋 특징</h4>
                    <ul class="feature-list">
                        <li>Google에서 개발한 FaceNet 사용</li>
                        <li>단순 리사이즈 후 특징 추출</li>
                        <li>빠른 처리 속도</li>
                        <li>일반적인 얼굴 인식 성능</li>
                    </ul>
                </div>
                
                <div class="model-card">
                    <h3>🚀 YOLOv8 + ArcFace</h3>
                    <p><strong>상태:</strong> {arcface_status}</p>
                    <p><strong>등록 사용자:</strong> {arcface_users}명</p>
                    <p><strong>임베딩 차원:</strong> 512차원</p>
                    <p><strong>유사도 측정:</strong> 코사인 유사도</p>
                    
                    <h4>📋 특징 (논문의 핵심)</h4>
                    <ul class="feature-list">
                        <li>dlib 기반 얼굴 정렬</li>
                        <li>랜드마크 기반 회전 보정</li>
                        <li>눈, 코, 입 위치 표준화</li>
                        <li>모든 이미지 동일 참조점 통일</li>
                    </ul>
                </div>
            </div>
            
            <div class="vs">⚡ VS ⚡</div>
            
            <div class="comparison-section">
                <h3>📊 논문 결과 vs 실제 테스트</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <h4>📄 논문 결과</h4>
                        <p><strong>FaceNet:</strong> 85.90%</p>
                        <p><strong>ArcFace:</strong> 96.09%</p>
                        <p class="highlight">+10.19% 우위</p>
                    </div>
                    <div class="stat-item">
                        <h4>⏱️ 처리 시간 (논문)</h4>
                        <p><strong>FaceNet:</strong> 0.41초</p>
                        <p><strong>ArcFace:</strong> 1.29초</p>
                        <p class="highlight">3.1배 차이</p>
                    </div>
                    <div class="stat-item">
                        <h4>🧪 총 테스트</h4>
                        <p><strong>완료:</strong> {len(comparison_results)}건</p>
                        <p><strong>벤치마크:</strong> {len(benchmark_history)}회</p>
                    </div>
                    <div class="stat-item">
                        <h4>🎯 예상 결과</h4>
                        <p>ArcFace가 정확도에서 우위</p>
                        <p>FaceNet이 속도에서 우위</p>
                    </div>
                </div>
            </div>
            
            <div class="api-section">
                <h3>🔗 API 엔드포인트</h3>
                
                <div class="endpoint">
                    <strong>POST /compare</strong> - 단일 이미지로 두 모델 비교
                    <br><small>업로드한 이미지를 FaceNet과 ArcFace로 동시 처리하여 결과 비교</small>
                </div>
                
                <div class="endpoint">
                    <strong>POST /benchmark</strong> - 벤치마크 테스트 실행
                    <br><small>등록된 모든 사용자 데이터로 대규모 성능 비교</small>
                </div>
                
                <div class="endpoint">
                    <strong>GET /results</strong> - 비교 결과 조회
                    <br><small>지금까지의 모든 비교 결과와 통계 데이터</small>
                </div>
                
                <div class="endpoint">
                    <strong>GET /model_info</strong> - 모델 상세 정보
                    <br><small>로드된 두 모델의 구성과 성능 정보</small>
                </div>
                
                <div class="endpoint">
                    <strong>GET /docs</strong> - Swagger API 문서
                    <br><small>대화형 API 문서 (테스트 가능)</small>
                </div>
            </div>
            
            <div class="comparison-section">
                <h3>💡 사용 방법</h3>
                <ol style="line-height: 1.8;">
                    <li><strong>단일 비교:</strong> POST /compare로 이미지 업로드하여 실시간 비교</li>
                    <li><strong>벤치마크:</strong> POST /benchmark로 전체 데이터셋 성능 측정</li>
                    <li><strong>결과 분석:</strong> GET /results로 통계 데이터 확인</li>
                    <li><strong>논문 검증:</strong> 실제 결과와 논문 결과 비교 분석</li>
                </ol>
            </div>
            
            <div class="comparison-section">
                <h3>🔬 논문의 핵심 가설</h3>
                <p><strong>"ArcFace outperforms other methods because it not only uses MTCNN but also adjusts face image to be in a straight direction as well as fixes the positions of eyebrows, eyes nose, and mouth so that all images have similar references."</strong></p>
                
                <p>우리 구현에서는:</p>
                <ul class="feature-list">
                    <li><strong>검출:</strong> MTCNN 대신 YOLOv8-Face 사용</li>
                    <li><strong>정렬:</strong> dlib으로 얼굴 회전 보정 구현</li>
                    <li><strong>표준화:</strong> 랜드마크 기반 특징점 통일</li>
                    <li><strong>비교:</strong> 논문과 동일한 방식으로 성능 측정</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(html_content)

@app.post("/compare")
async def compare_models(file: UploadFile = File(...)):
    """단일 이미지로 FaceNet vs ArcFace 성능 비교"""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    if not facenet_model or not arcface_model:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        # 이미지 로드
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        logger.info(f"🔍 이미지 비교 시작: {file.filename}")
        
        # 두 모델 병렬 처리
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            facenet_future = executor.submit(facenet_model.process_image, image_np)
            arcface_future = executor.submit(arcface_model.process_image, image_np)
            
            facenet_result = facenet_future.result()
            arcface_result = arcface_future.result()
        
        # 결과 비교 분석
        comparison = analyze_comparison(facenet_result, arcface_result)
        
        # 결과 저장
        comparison_data = {
            'id': len(comparison_results) + 1,
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            'facenet_result': facenet_result,
            'arcface_result': arcface_result,
            'comparison': comparison
        }
        
        comparison_results.append(comparison_data)
        
        # 백그라운드에서 저장
        save_results()
        
        logger.info(f"✅ 비교 완료: {comparison['winner']}가 우수")
        
        return {
            'success': True,
            'comparison_id': comparison_data['id'],
            'facenet': facenet_result,
            'arcface': arcface_result,
            'comparison_analysis': comparison,
            'image_info': {
                'filename': file.filename,
                'size': len(image_data),
                'dimensions': f"{image.width}x{image.height}"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 비교 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 비교 중 오류: {str(e)}")

def analyze_comparison(facenet_result: Dict, arcface_result: Dict) -> Dict:
    """두 모델 결과 비교 분석"""
    analysis = {
        'detection_comparison': {},
        'recognition_comparison': {},
        'performance_comparison': {},
        'winner': None,
        'winner_reason': None
    }
    
    # 1. 검출 성능 비교
    fn_faces = facenet_result.get('faces_found', 0)
    af_faces = arcface_result.get('faces_found', 0)
    
    analysis['detection_comparison'] = {
        'facenet_faces': fn_faces,
        'arcface_faces': af_faces,
        'detection_winner': 'facenet' if fn_faces > af_faces else 'arcface' if af_faces > fn_faces else 'tie'
    }
    
    # 2. 인식 성능 비교
    if facenet_result.get('success') and arcface_result.get('success'):
        fn_main = facenet_result.get('main_face', {})
        af_main = arcface_result.get('main_face', {})
        
        fn_recognized = fn_main.get('recognized', False)
        af_recognized = af_main.get('recognized', False)
        fn_confidence = fn_main.get('confidence', 0)
        af_confidence = af_main.get('confidence', 0)
        
        analysis['recognition_comparison'] = {
            'facenet': {
                'recognized': fn_recognized,
                'user': fn_main.get('user_name', 'Unknown'),
                'confidence': fn_confidence,
                'distance_metric': fn_main.get('distance', 'N/A')
            },
            'arcface': {
                'recognized': af_recognized,
                'user': af_main.get('user_name', 'Unknown'),
                'confidence': af_confidence,
                'similarity_metric': af_main.get('similarity', 'N/A')
            }
        }
        
        # 인식 성공률 비교
        if fn_recognized and af_recognized:
            recognition_winner = 'arcface' if af_confidence > fn_confidence else 'facenet'
        elif af_recognized:
            recognition_winner = 'arcface'
        elif fn_recognized:
            recognition_winner = 'facenet'
        else:
            recognition_winner = 'tie'
        
        analysis['recognition_comparison']['winner'] = recognition_winner
    
    # 3. 성능 비교
    fn_perf = facenet_result.get('performance', {})
    af_perf = arcface_result.get('performance', {})
    
    fn_total = fn_perf.get('total_time', float('inf'))
    af_total = af_perf.get('total_time', float('inf'))
    
    analysis['performance_comparison'] = {
        'facenet': {
            'total_time': fn_total,
            'detection_time': fn_perf.get('detection_time', 0),
            'recognition_time': fn_perf.get('recognition_time', 0)
        },
        'arcface': {
            'total_time': af_total,
            'detection_time': af_perf.get('detection_time', 0),
            'alignment_time': af_perf.get('alignment_time', 0),
            'recognition_time': af_perf.get('recognition_time', 0)
        },
        'speed_winner': 'facenet' if fn_total < af_total else 'arcface',
        'speed_advantage': f"{abs(fn_total - af_total):.3f}초 차이"
    }
    
    # 4. 최종 승자 결정
    recognition_winner = analysis['recognition_comparison'].get('winner', 'tie')
    speed_winner = analysis['performance_comparison']['speed_winner']
    
    if recognition_winner == 'arcface':
        analysis['winner'] = 'arcface'
        analysis['winner_reason'] = '높은 인식 정확도 (논문 예측과 일치)'
    elif recognition_winner == 'facenet':
        analysis['winner'] = 'facenet'
        analysis['winner_reason'] = '빠른 처리 속도와 인식 성공'
    else:
        analysis['winner'] = speed_winner
        analysis['winner_reason'] = '인식 성능 동일, 처리 속도 우위'
    
    return analysis

@app.post("/benchmark")
async def run_benchmark(background_tasks: BackgroundTasks):
    """전체 데이터셋으로 벤치마크 테스트"""
    
    if not facenet_model or not arcface_model:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        logger.info("🧪 벤치마크 테스트 시작")
        
        # 백그라운드에서 벤치마크 실행
        background_tasks.add_task(execute_benchmark)
        
        return {
            'message': '벤치마크 테스트가 백그라운드에서 시작되었습니다',
            'status': 'started',
            'estimated_time': '1-5분 (데이터 크기에 따라 달라짐)',
            'check_progress': 'GET /benchmark/status로 진행상황 확인'
        }
        
    except Exception as e:
        logger.error(f"❌ 벤치마크 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=f"벤치마크 실행 오류: {str(e)}")

async def execute_benchmark():
    """벤치마크 실행 (백그라운드 작업)"""
    global benchmark_history
    
    try:
        start_time = time.time()
        
        # 테스트 데이터 준비
        test_images = []
        users_dir = "data/users"
        
        if not os.path.exists(users_dir):
            logger.error("❌ 사용자 데이터 폴더 없음")
            return
        
        # 각 사용자별 이미지 수집
        for user_name in os.listdir(users_dir):
            user_path = os.path.join(users_dir, user_name)
            if os.path.isdir(user_path):
                for img_file in os.listdir(user_path)[:5]:  # 사용자당 최대 5장
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append({
                            'path': os.path.join(user_path, img_file),
                            'true_user': user_name,
                            'filename': img_file
                        })
        
        logger.info(f"📊 총 {len(test_images)}장의 테스트 이미지로 벤치마크 시작")
        
        facenet_results = []
        arcface_results = []
        
        # 각 이미지에 대해 두 모델 테스트
        for i, test_data in enumerate(test_images):
            try:
                image = cv2.imread(test_data['path'])
                if image is None:
                    continue
                
                # 병렬 처리
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    fn_future = executor.submit(facenet_model.process_image, image)
                    af_future = executor.submit(arcface_model.process_image, image)
                    
                    fn_result = fn_future.result()
                    af_result = af_future.result()
                
                # 결과 저장
                facenet_results.append({
                    'true_user': test_data['true_user'],
                    'result': fn_result,
                    'filename': test_data['filename']
                })
                
                arcface_results.append({
                    'true_user': test_data['true_user'],
                    'result': af_result,
                    'filename': test_data['filename']
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"📈 진행률: {i+1}/{len(test_images)} ({(i+1)/len(test_images)*100:.1f}%)")
                
            except Exception as e:
                logger.warning(f"⚠️ {test_data['filename']} 처리 실패: {e}")
                continue
        
        # 성능 분석
        facenet_metrics = calculate_metrics(facenet_results, "FaceNet")
        arcface_metrics = calculate_metrics(arcface_results, "ArcFace")
        
        # 벤치마크 결과 저장
        benchmark_result = {
            'id': len(benchmark_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'test_duration': round(time.time() - start_time, 2),
            'total_images': len(test_images),
            'processed_images': len(facenet_results),
            'facenet_metrics': facenet_metrics,
            'arcface_metrics': arcface_metrics,
            'comparison': {
                'accuracy_winner': 'ArcFace' if arcface_metrics['accuracy'] > facenet_metrics['accuracy'] else 'FaceNet',
                'speed_winner': 'FaceNet' if facenet_metrics['avg_processing_time'] < arcface_metrics['avg_processing_time'] else 'ArcFace',
                'accuracy_difference': abs(arcface_metrics['accuracy'] - facenet_metrics['accuracy']),
                'speed_difference': abs(arcface_metrics['avg_processing_time'] - facenet_metrics['avg_processing_time'])
            }
        }
        
        benchmark_history.append(benchmark_result)
        save_results()
        
        logger.info(f"✅ 벤치마크 완료!")
        logger.info(f"   - FaceNet 정확도: {facenet_metrics['accuracy']:.2f}%")
        logger.info(f"   - ArcFace 정확도: {arcface_metrics['accuracy']:.2f}%")
        logger.info(f"   - 처리 시간: {benchmark_result['test_duration']}초")
        
    except Exception as e:
        logger.error(f"❌ 벤치마크 실행 오류: {e}")

def calculate_metrics(results: List[Dict], model_name: str) -> Dict:
    """성능 지표 계산"""
    if not results:
        return {
            'model_name': model_name,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'avg_processing_time': 0.0,
            'total_tests': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0
        }
    
    total_tests = len(results)
    correct_recognitions = 0
    total_processing_time = 0
    recognition_times = []
    
    # TP, FP, TN, FN 계산을 위한 변수
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for result_data in results:
        true_user = result_data['true_user']
        result = result_data['result']
        
        if result.get('success', False):
            main_face = result.get('main_face', {})
            recognized = main_face.get('recognized', False)
            predicted_user = main_face.get('user_name', '')
            
            # 처리 시간 수집
            perf = result.get('performance', {})
            total_time = perf.get('total_time', 0)
            total_processing_time += total_time
            recognition_times.append(total_time)
            
            # 정확도 계산
            if recognized and predicted_user == true_user:
                correct_recognitions += 1
                true_positives += 1
            elif recognized and predicted_user != true_user:
                false_positives += 1
            elif not recognized:
                false_negatives += 1
    
    # 지표 계산
    accuracy = (correct_recognitions / total_tests * 100) if total_tests > 0 else 0
    precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    avg_processing_time = (total_processing_time / len(recognition_times)) if recognition_times else 0
    
    return {
        'model_name': model_name,
        'accuracy': round(accuracy, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1_score * 100, 2),
        'avg_processing_time': round(avg_processing_time, 3),
        'total_tests': total_tests,
        'successful_recognitions': correct_recognitions,
        'failed_recognitions': total_tests - correct_recognitions,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

@app.get("/benchmark/status")
async def get_benchmark_status():
    """벤치마크 진행 상황 조회"""
    if benchmark_history:
        latest = benchmark_history[-1]
        return {
            'status': 'completed',
            'latest_benchmark': latest,
            'total_benchmarks': len(benchmark_history)
        }
    else:
        return {
            'status': 'no_benchmarks',
            'message': '아직 실행된 벤치마크가 없습니다'
        }

@app.get("/results")
async def get_results(limit: int = Query(50, description="조회할 결과 수")):
    """비교 결과 조회"""
    try:
        # 최신 결과부터 반환
        recent_comparisons = comparison_results[-limit:] if comparison_results else []
        recent_benchmarks = benchmark_history[-5:] if benchmark_history else []
        
        # 통계 계산
        total_comparisons = len(comparison_results)
        facenet_wins = sum(1 for r in comparison_results if r['comparison']['winner'] == 'facenet')
        arcface_wins = sum(1 for r in comparison_results if r['comparison']['winner'] == 'arcface')
        
        return {
            'summary': {
                'total_comparisons': total_comparisons,
                'total_benchmarks': len(benchmark_history),
                'facenet_wins': facenet_wins,
                'arcface_wins': arcface_wins,
                'win_rate': {
                    'facenet': round(facenet_wins / total_comparisons * 100, 2) if total_comparisons > 0 else 0,
                    'arcface': round(arcface_wins / total_comparisons * 100, 2) if total_comparisons > 0 else 0
                }
            },
            'recent_comparisons': recent_comparisons,
            'recent_benchmarks': recent_benchmarks,
            'paper_comparison': {
                'paper_results': {
                    'facenet_accuracy': 85.90,
                    'arcface_accuracy': 96.09,
                    'facenet_time': 0.41,
                    'arcface_time': 1.29
                },
                'our_results': recent_benchmarks[-1] if recent_benchmarks else None
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 결과 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"결과 조회 중 오류: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """로드된 모델들의 상세 정보"""
    try:
        facenet_info = facenet_model.get_model_info() if facenet_model else None
        arcface_info = arcface_model.get_model_info() if arcface_model else None
        
        return {
            'facenet': facenet_info,
            'arcface': arcface_info,
            'comparison_setup': {
                'detection_model': 'YOLOv8-Face (공통)',
                'facenet_recognition': 'FaceNet InceptionResNetV1',
                'arcface_recognition': 'ArcFace ResNet50 + Face Alignment',
                'key_differences': [
                    'FaceNet: 단순 리사이즈 후 특징 추출',
                    'ArcFace: dlib 랜드마크 기반 얼굴 정렬',
                    'FaceNet: 유클리드 거리 사용',
                    'ArcFace: 코사인 유사도 사용'
                ]
            },
            'paper_hypothesis': {
                'claim': 'ArcFace가 얼굴 정렬과 특징점 표준화로 더 높은 정확도 달성',
                'expected_result': 'ArcFace > FaceNet (정확도), FaceNet > ArcFace (속도)'
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 모델 정보 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"모델 정보 조회 중 오류: {str(e)}")

@app.post("/reload_models")
async def reload_models():
    """모델 다시 로드"""
    global facenet_model, arcface_model
    
    try:
        logger.info("🔄 모델 다시 로드 시작...")
        
        # 기존 모델 해제
        facenet_model = None
        arcface_model = None
        
        # 새로 로드
        facenet_model = load_facenet_model()
        arcface_model = load_arcface_model()
        
        # 사용자 데이터 다시 로드
        if facenet_model:
            if not facenet_model.load_embeddings():
                facenet_model.load_user_database()
        
        if arcface_model:
            if not arcface_model.load_embeddings():
                arcface_model.load_user_database()
        
        return {
            'success': True,
            'message': '모델이 성공적으로 다시 로드되었습니다',
            'facenet_loaded': facenet_model is not None,
            'arcface_loaded': arcface_model is not None,
            'facenet_users': len(facenet_model.user_names) if facenet_model else 0,
            'arcface_users': len(arcface_model.user_names) if arcface_model else 0
        }
        
    except Exception as e:
        logger.error(f"❌ 모델 다시 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 로드 오류: {str(e)}")

@app.get("/stats")
async def get_detailed_stats():
    """상세 통계 정보"""
    try:
        if not comparison_results and not benchmark_history:
            return {
                'message': '아직 수집된 데이터가 없습니다',
                'total_tests': 0
            }
        
        # 비교 결과 분석
        comparison_stats = {
            'total_comparisons': len(comparison_results),
            'winners': {
                'facenet': 0,
                'arcface': 0,
                'tie': 0
            },
            'avg_processing_times': {
                'facenet': [],
                'arcface': []
            },
            'recognition_rates': {
                'facenet': {'success': 0, 'fail': 0},
                'arcface': {'success': 0, 'fail': 0}
            }
        }
        
        for result in comparison_results:
            # 승자 통계
            winner = result['comparison']['winner']
            comparison_stats['winners'][winner] = comparison_stats['winners'].get(winner, 0) + 1
            
            # 처리 시간 통계
            fn_time = result['facenet_result'].get('performance', {}).get('total_time', 0)
            af_time = result['arcface_result'].get('performance', {}).get('total_time', 0)
            
            comparison_stats['avg_processing_times']['facenet'].append(fn_time)
            comparison_stats['avg_processing_times']['arcface'].append(af_time)
            
            # 인식 성공률 통계
            fn_success = result['facenet_result'].get('main_face', {}).get('recognized', False)
            af_success = result['arcface_result'].get('main_face', {}).get('recognized', False)
            
            comparison_stats['recognition_rates']['facenet']['success' if fn_success else 'fail'] += 1
            comparison_stats['recognition_rates']['arcface']['success' if af_success else 'fail'] += 1
        
        # 평균 처리 시간 계산
        facenet_times = comparison_stats['avg_processing_times']['facenet']
        arcface_times = comparison_stats['avg_processing_times']['arcface']
        
        comparison_stats['avg_processing_times'] = {
            'facenet': round(sum(facenet_times) / len(facenet_times), 3) if facenet_times else 0,
            'arcface': round(sum(arcface_times) / len(arcface_times), 3) if arcface_times else 0
        }
        
        # 최신 벤치마크 결과
        latest_benchmark = benchmark_history[-1] if benchmark_history else None
        
        return {
            'comparison_stats': comparison_stats,
            'benchmark_stats': {
                'total_benchmarks': len(benchmark_history),
                'latest_benchmark': latest_benchmark
            },
            'paper_vs_reality': {
                'paper_claim': 'ArcFace 96.09% vs FaceNet 85.90%',
                'our_results': {
                    'facenet_accuracy': latest_benchmark['facenet_metrics']['accuracy'] if latest_benchmark else 'N/A',
                    'arcface_accuracy': latest_benchmark['arcface_metrics']['accuracy'] if latest_benchmark else 'N/A'
                } if latest_benchmark else 'No benchmark data'
            },
            'insights': generate_insights(comparison_stats, latest_benchmark)
        }
        
    except Exception as e:
        logger.error(f"❌ 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류: {str(e)}")

def generate_insights(comparison_stats: Dict, latest_benchmark: Optional[Dict]) -> List[str]:
    """통계 기반 인사이트 생성"""
    insights = []
    
    # 승률 분석
    total_comparisons = comparison_stats['total_comparisons']
    if total_comparisons > 0:
        arcface_wins = comparison_stats['winners'].get('arcface', 0)
        facenet_wins = comparison_stats['winners'].get('facenet', 0)
        
        arcface_win_rate = arcface_wins / total_comparisons * 100
        facenet_win_rate = facenet_wins / total_comparisons * 100
        
        if arcface_win_rate > facenet_win_rate:
            insights.append(f"🏆 ArcFace가 {arcface_win_rate:.1f}%의 승률로 우세 (논문 예측과 일치)")
        elif facenet_win_rate > arcface_win_rate:
            insights.append(f"🏆 FaceNet이 {facenet_win_rate:.1f}%의 승률로 우세 (논문 예측과 다름)")
        else:
            insights.append("🤝 두 모델이 비슷한 성능을 보임")
    
    # 처리 시간 분석
    fn_time = comparison_stats['avg_processing_times']['facenet']
    af_time = comparison_stats['avg_processing_times']['arcface']
    
    if fn_time > 0 and af_time > 0:
        if fn_time < af_time:
            speed_advantage = round((af_time - fn_time) / fn_time * 100, 1)
            insights.append(f"⚡ FaceNet이 {speed_advantage}% 더 빠름 (논문과 일치)")
        else:
            speed_advantage = round((fn_time - af_time) / af_time * 100, 1)
            insights.append(f"⚡ ArcFace가 {speed_advantage}% 더 빠름 (논문과 다름)")
    
    # 벤치마크 결과 분석
    if latest_benchmark:
        fn_accuracy = latest_benchmark['facenet_metrics']['accuracy']
        af_accuracy = latest_benchmark['arcface_metrics']['accuracy']
        
        if af_accuracy > fn_accuracy:
            accuracy_diff = af_accuracy - fn_accuracy
            insights.append(f"🎯 ArcFace가 {accuracy_diff:.1f}%p 더 정확함")
            
            # 논문과 비교
            paper_diff = 96.09 - 85.90  # 논문의 차이
            if accuracy_diff > paper_diff * 0.5:
                insights.append("📄 논문의 결과와 유사한 패턴 확인")
            else:
                insights.append("📄 논문보다 작은 차이, 하지만 동일한 경향")
        else:
            insights.append("🤔 FaceNet이 더 정확함 (논문 결과와 다름)")
    
    # 인식 성공률 분석
    fn_recognition = comparison_stats['recognition_rates']['facenet']
    af_recognition = comparison_stats['recognition_rates']['arcface']
    
    fn_success_rate = fn_recognition['success'] / (fn_recognition['success'] + fn_recognition['fail']) * 100 if (fn_recognition['success'] + fn_recognition['fail']) > 0 else 0
    af_success_rate = af_recognition['success'] / (af_recognition['success'] + af_recognition['fail']) * 100 if (af_recognition['success'] + af_recognition['fail']) > 0 else 0
    
    if af_success_rate > fn_success_rate:
        insights.append(f"✅ ArcFace 인식률 {af_success_rate:.1f}% > FaceNet {fn_success_rate:.1f}%")
    elif fn_success_rate > af_success_rate:
        insights.append(f"✅ FaceNet 인식률 {fn_success_rate:.1f}% > ArcFace {af_success_rate:.1f}%")
    
    return insights

@app.get("/health")
async def health_check():
    """서버 상태 체크"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'facenet_loaded': facenet_model is not None,
            'arcface_loaded': arcface_model is not None
        },
        'data_stats': {
            'total_comparisons': len(comparison_results),
            'total_benchmarks': len(benchmark_history)
        },
        'version': '1.0.0'
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🎯 YOLOv8 + FaceNet vs ArcFace 성능 비교 시스템")
    print("="*80)
    print("📋 기능:")
    print("  • 실시간 두 모델 성능 비교")
    print("  • 대규모 벤치마크 테스트")
    print("  • 논문 결과 vs 실제 결과 분석")
    print("  • 상세 통계 및 인사이트 제공")
    print("\n📄 논문 가설:")
    print('  "ArcFace outperforms other methods because it not only uses MTCNN')
    print('   but also adjusts face image to be in a straight direction as well')
    print('   as fixes the positions of eyebrows, eyes nose, and mouth')
    print('   so that all images have similar references."')
    print("\n🔧 우리 구현:")
    print("  • 검출: YOLOv8-Face (MTCNN 대신)")
    print("  • FaceNet: 단순 리사이즈 → 특징 추출")
    print("  • ArcFace: dlib 정렬 → 특징점 표준화 → 특징 추출")
    print("\n🌐 웹 인터페이스: http://localhost:8002")
    print("📚 API 문서: http://localhost:8002/docs")
    print("="*80 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8002,
        log_level="info",
        access_log=True
    )