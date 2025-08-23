from fastapi import FastAPI, File, UploadFile, HTTPException
import face_recognition
import numpy as np
from PIL import Image
import io
import os
import pickle

app = FastAPI()

# 연예인 얼굴 데이터 저장할 딕셔너리
known_face_encodings = []
known_face_names = []

# 연예인 데이터 로드 함수
def load_celebrity_data():
    """연예인 얼굴 데이터를 로드합니다."""
    global known_face_encodings, known_face_names
    
    # data/celebrities 폴더가 있는지 확인
    celebrities_dir = "data/celebrities"
    if not os.path.exists(celebrities_dir):
        print("연예인 데이터 폴더가 없습니다. 샘플 데이터를 생성하세요.")
        return
    
    # 각 연예인 폴더별로 처리
    for celebrity_name in os.listdir(celebrities_dir):
        celebrity_path = os.path.join(celebrities_dir, celebrity_name)
        if os.path.isdir(celebrity_path):
            print(f"{celebrity_name} 데이터 로드 중...")
            
            # 해당 연예인의 모든 이미지 처리
            for image_file in os.listdir(celebrity_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(celebrity_path, image_file)
                    
                    # 얼굴 인식
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(celebrity_name)
                        print(f"  - {image_file} 처리 완료")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 연예인 데이터 로드"""
    load_celebrity_data()
    print(f"총 {len(known_face_encodings)}개의 연예인 얼굴 데이터를 로드했습니다.")

@app.get("/")
def read_root():
    return {
        "message": "연예인 얼굴 인식 서버가 실행 중입니다!",
        "loaded_celebrities": len(known_face_encodings),
        "celebrity_names": list(set(known_face_names))
    }

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """업로드된 이미지에서 얼굴을 인식합니다."""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 이미지 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # 얼굴 위치와 인코딩 찾기
        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        
        if not face_encodings:
            return {"message": "얼굴을 찾을 수 없습니다.", "recognized": False}
        
        results = []
        
        for face_encoding in face_encodings:
            # 알려진 얼굴과 비교
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if True in matches:
                # 가장 유사한 얼굴 찾기
                best_match_index = np.argmin(distances)
                confidence = 1 - distances[best_match_index]  # 신뢰도 계산
                
                if matches[best_match_index] and confidence > 0.6:  # 신뢰도 60% 이상
                    celebrity_name = known_face_names[best_match_index]
                    results.append({
                        "celebrity": celebrity_name,
                        "confidence": round(confidence * 100, 2),
                        "recognized": True
                    })
                else:
                    results.append({
                        "celebrity": "알 수 없음",
                        "confidence": 0,
                        "recognized": False
                    })
            else:
                results.append({
                    "celebrity": "알 수 없음", 
                    "confidence": 0,
                    "recognized": False
                })
        
        return {
            "faces_found": len(face_encodings),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 인식 중 오류: {str(e)}")

@app.get("/celebrities")
def get_celebrities():
    """등록된 연예인 목록을 반환합니다."""
    unique_names = list(set(known_face_names))
    return {
        "total_count": len(known_face_encodings),
        "unique_celebrities": len(unique_names),
        "celebrities": unique_names
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)