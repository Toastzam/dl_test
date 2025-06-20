from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from contextlib import asynccontextmanager
import io
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.cm as cm
import base64
import os

# --- 모델 관련 (사용자님의 실제 SimCLR 인코더 클래스로 교체 필요) ---
# 이 클래스는 사용자님의 SimCLR 인코더와 동일해야 합니다.
# 실제 프로젝트에서는 별도의 파일로 빼서 import 하는 것이 좋습니다.
import torch.nn as nn
from transformers import ViTConfig, ViTModel, ViTImageProcessor # ViTImageProcessor 추가

class SimCLREncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", projection_dim=128):
        super().__init__()
        config = ViTConfig.from_pretrained(model_name)
        # ViTModel을 사용하면, 내부적으로 어텐션 가중치를 추출하기 위한 hook을 걸 수 있습니다.
        # 실제 어텐션 맵 시각화를 위해서는 더 복잡한 로직이 필요합니다.
        self.vit = ViTModel.from_pretrained(model_name, output_attentions=True) # 어텐션 가중치 출력 활성화
        self.projection_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, projection_dim)
        )
        self.model_name = model_name

    def forward(self, pixel_values):
        # ViT의 마지막 hidden state의 CLS 토큰만 사용
        # output_attentions=True 설정하면 outputs.attentions 에 어텐션 가중치도 반환됨
        outputs = self.vit(pixel_values=pixel_values)
        vit_output = outputs.last_hidden_state[:, 0, :] # CLS token output
        projection = self.projection_head(vit_output)
        return projection, outputs.attentions # 어텐션 가중치도 함께 반환

# --- FastAPI 앱 초기화 ---

# --- 전역 변수 설정 ---
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 서버 시작 시 모델 로드 (앱이 처음 실행될 때 한 번만) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    print(f"모델 로드 중... (Device: {device})")
    try:
        model = SimCLREncoder(projection_dim=128)
        # 학습된 모델 가중치 파일 경로
        model_weights_path = "backend/static/simclr_dog_encoder_epoch_50.pth"

        if os.path.exists(model_weights_path):
            print(f"학습된 모델 가중치 파일 로드: {model_weights_path}")
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
        else:
            print("학습된 모델 가중치 파일을 찾을 수 없습니다. 모델이 무작위 초기화 상태로 사용됩니다.")
            print(f"경로 확인: {os.path.abspath(model_weights_path)}")

        model.eval() # 추론 모드로 설정 (드롭아웃, 배치 정규화 등에 영향)
        model.to(device)

        # ViTImageProcessor는 이미지 전처리에 사용됩니다.
        processor = ViTImageProcessor.from_pretrained(model.model_name)
        print("모델 로드 및 프로세서 준비 완료.")

    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        # 오류 발생 시 모델을 None으로 유지하여 API 호출 시 처리할 수 있도록 함
        model = None
        processor = None
    yield
    print("FastAPI 앱 종료 중...")
        
app = FastAPI(lifespan=lifespan)

# 정적 파일 (이미지) 서빙 설정: 'static' 폴더의 파일을 '/static' 경로로 접근 가능하게 함
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 히트맵 생성 함수 (매우 간략화된 예시) ---
# 실제 어텐션 맵 시각화를 위해서는 ViTModel의 outputs.attentions를 사용하여
# 각 헤드의 어텐션 가중치를 조합하고, 이를 이미지 크기에 맞게 업샘플링해야 합니다.
# 이는 복잡한 작업이므로, 현재는 시각화 효과를 위한 "더미" 히트맵을 생성합니다.
def generate_dummy_heatmap_b64(image_pil: Image.Image, value=0.5):
    # 입력 이미지와 동일한 크기의 히트맵 데이터를 생성 (0.0 ~ 1.0)
    heatmap_data = np.full((image_pil.height, image_pil.width), value, dtype=np.float32)

    # 중앙에 밝은 원을 그리는 예시 (실제 어텐션 맵이 아님)
    center_x, center_y = image_pil.width // 2, image_pil.height // 2
    radius = min(image_pil.width, image_pil.height) // 4
    cv2.circle(heatmap_data, (center_x, center_y), radius, 1.0, -1) # 원형으로 강조

    # 히트맵 데이터를 컬러맵으로 변환 (matplotlib의 jet 컬러맵 사용)
    # RGBA (Red, Green, Blue, Alpha) 형태로 나옴
    heatmap_colored = cm.jet(heatmap_data)
    # 0-1 스케일의 RGBA를 0-255 스케일의 RGBA로 변환
    heatmap_colored_uint8 = (heatmap_colored * 255).astype(np.uint8)

    # PIL Image 객체로 변환
    heatmap_pil = Image.fromarray(heatmap_colored_uint8, 'RGBA')

    # PIL Image를 Bytes로 변환 (PNG 형식)
    buffer = io.BytesIO()
    heatmap_pil.save(buffer, format="PNG")
    heatmap_bytes = buffer.getvalue()
    return base64.b64encode(heatmap_bytes).decode('utf-8')

# 기존 generate_dummy_heatmap_b64는 임시로 남겨두고,
# 실제 어텐션 맵을 위한 함수를 새로 만듭니다.
# 주의: 이 코드는 개념 설명이며, 실제 동작하는 완전한 코드는 아닙니다.
# ViT의 내부 구조와 attention map 추출 방법에 대한 깊은 이해가 필요합니다.
def get_attention_heatmap(original_image_pil: Image.Image, attentions, patch_size=16):
    # attentions는 outputs.attentions 입니다.
    # 일반적으로 ViT의 attentions는 (num_layers, batch_size, num_heads, seq_len, seq_len) 형태입니다.
    # 우리는 마지막 레이어의 어텐션 가중치를 사용하고,
    # 특히 [CLS] 토큰이 다른 패치들에 얼마나 집중했는지를 시각화하는 것이 일반적입니다.

    # 1. CLS 토큰에 대한 어텐션 가중치 추출 (예시)
    # outputs.attentions는 튜플 형태이므로, 마지막 레이어 ([0] 인덱스)를 가져옵니다.
    # attentions_last_layer = attentions[-1] # (batch_size, num_heads, seq_len, seq_len)

    # 모든 헤드의 CLS 토큰 (인덱스 0)이 다른 패치들 (인덱스 1부터)에 대한 평균 어텐션 가중치를 계산
    # 예: attention_weights = attentions_last_layer[0, :, 0, 1:].mean(dim=0)
    # (이 부분은 모델의 정확한 구조와 원하는 시각화 방식에 따라 달라집니다.)
    
    # -------------------------------------------------------------
    # 여기서는 임시로 ViT 모델 내부의 Attention 가중치를 직접 접근하는 예시를 보여드립니다.
    # 실제로는 model.vit.encoder.layer[-1].attention.self.get_attn_map() 같은
    # 특정 hook이나 메서드를 통해 가져오는 것이 더 안정적일 수 있습니다.
    # DINO 모델 같은 경우는 공식 깃허브에 시각화 코드가 잘 제공됩니다.
    # -------------------------------------------------------------

    # **가장 간단하게 ViT에서 어텐션 맵을 얻는 방법 중 하나 (개념적 코드):**
    # ViTModel.forward 시 output_attentions=True 설정하면 outputs.attentions가 반환됩니다.
    # outputs.attentions는 각 레이어의 어텐션 텐서들의 튜플입니다.
    # 가장 마지막 레이어의 어텐션 가중치 (outputs.attentions[-1])를 사용합니다.
    # 이 가중치는 (batch_size, num_heads, sequence_length, sequence_length) 형태입니다.
    # 여기서 sequence_length는 1 (CLS 토큰) + (이미지 패치 수) 입니다.
    
    # CLS 토큰 (인덱스 0)이 다른 패치들에 대한 어텐션 가중치만 가져옵니다.
    # 모든 헤드의 평균을 사용하는 것이 일반적입니다.
    # 예시: attention_map = attentions[-1][0, :, 0, 1:].mean(dim=0) # [num_patches]
    
    # 지금은 실제 어텐션 가중치 추출 로직 대신, '더미'로 대체합니다.
    # 이 부분을 직접 구현하셔야 합니다!
    
    # **임시로, 다시 더미 히트맵을 생성하여 전달합니다.**
    # (사용자님이 이 함수를 실제 어텐션 맵 생성 로직으로 교체해야 함)
    # -----------------------------------------------------------------
    
    # 실제 어텐션 맵 데이터 (0~1 사이)
    # 이 부분에 위에서 계산한 `attention_map`을 사용해야 합니다.
    # attention_map = attention_map.cpu().numpy().reshape(image_width // patch_size, image_height // patch_size)
    # 이후 resize하여 원본 이미지 크기로 맞춤

    # 현재는 아래 generate_dummy_heatmap_b64를 계속 사용합니다.
    # 따라서 계속 중앙에 원형 히트맵이 나올 것입니다.
    return generate_dummy_heatmap_b64(original_image_pil) # <<< 이 부분을 실제 로직으로 교체해야 함!


def get_attention_heatmap(original_image_pil: Image.Image, attentions, patch_size=16):
    heatmap_data = np.full((original_image_pil.height, original_image_pil.width), 0.5, dtype=np.float32)
    center_x, center_y = original_image_pil.width // 2, original_image_pil.height // 2
    radius = min(original_image_pil.width, original_image_pil.height) // 4
    cv2.circle(heatmap_data, (center_x, center_y), radius, 1.0, -1)
    heatmap_colored = cm.jet(heatmap_data)
    heatmap_colored_uint8 = (heatmap_colored * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored_uint8, 'RGBA')
    buffer = io.BytesIO()
    heatmap_pil.save(buffer, format="PNG")
    heatmap_bytes = buffer.getvalue()
    heatmap_b64 = base64.b64encode(heatmap_bytes).decode('utf-8')
    return heatmap_b64, heatmap_data


# --- API 엔드포인트 정의 ---
@app.post("/compare_dogs_with_heatmap/")
async def compare_dog_images_with_heatmap(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    if model is None or processor is None:
        return JSONResponse(status_code=503, content={"message": "모델이 아직 로드되지 않았거나 오류가 발생했습니다."})

    try:
        # 1. 이미지 읽기 (PIL Image 객체로)
        image1_pil = Image.open(io.BytesIO(await file1.read())).convert("RGB")
        image2_pil = Image.open(io.BytesIO(await file2.read())).convert("RGB")

        # 2. 이미지 전처리 및 모델 입력 준비
        # ViTImageProcessor를 사용하면 편리하게 전처리 가능
        inputs1 = processor(images=image1_pil, return_tensors="pt").to(device)
        inputs2 = processor(images=image2_pil, return_tensors="pt").to(device)

        # 3. 모델 순전파 (특징 추출 및 어텐션 가중치 얻기)
        with torch.no_grad():
            features1, attentions1 = model(inputs1['pixel_values'])
            features2, attentions2 = model(inputs2['pixel_values'])

        # 4. 유사도 계산
        # SimCLR의 projection_head 출력을 사용해야 하지만,
        # 유사도 시각화를 위해 feature(CLS token output)를 사용할 수도 있습니다.
        # 여기서는 features (CLS token output) 사용
        similarity = F.cosine_similarity(features1, features2).item()

        # 5. 히트맵 생성 및 Base64 인코딩
        # 실제 어텐션 맵을 사용하는 함수로 교체 필요!
        heatmap_b64_1, heatmap1 = get_attention_heatmap(image1_pil, attentions1)
        heatmap_b64_2, heatmap2 = get_attention_heatmap(image2_pil, attentions2)
        
        # 예시: 더미 좌표 (중앙)
        point1 = {"x": image1_pil.width // 2, "y": image1_pil.height // 2}
        point2 = {"x": image2_pil.width // 2, "y": image2_pil.height // 2}

        y1, x1 = np.unravel_index(np.argmax(heatmap1), heatmap1.shape)
        y2, x2 = np.unravel_index(np.argmax(heatmap2), heatmap2.shape)
        point1 = {"x": int(x1), "y": int(y1)}
        point2 = {"x": int(x2), "y": int(y2)}

        return JSONResponse({
            "similarity": similarity,
            "heatmap_image1": f"data:image/png;base64,{heatmap_b64_1}",
            "heatmap_image2": f"data:image/png;base64,{heatmap_b64_2}",
            "point1": point1,  # 첫 번째 이미지의 유사 부위 좌표
            "point2": point2,  # 두 번째 이미지의 유사 부위 좌표
            "message": "강아지 유사도 비교 및 히트맵 생성 완료!"
        })

    except Exception as e:
        print(f"API 처리 중 오류 발생: {e}")
        return JSONResponse(status_code=500, content={"message": f"서버 오류: {str(e)}"})