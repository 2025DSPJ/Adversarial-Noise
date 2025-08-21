from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms  
from transformers import AutoImageProcessor, AutoModelForImageClassification 
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage import gaussian_filter
import base64
import io
import os
import requests
import json

app = Flask(__name__)

# CORS 설정
CORS(app, origins=[
    "http://localhost:3000",  # React
    "http://localhost:8080",  # Vue
    "http://localhost:5173",  # Vite
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173"
])

# 모델 로드
print("🎨 WikiArt-Style 예술 분류 모델 로딩 중...")
art_processor = AutoImageProcessor.from_pretrained("prithivMLmods/WikiArt-Style")
art_model = AutoModelForImageClassification.from_pretrained("prithivMLmods/WikiArt-Style")
print("✅ WikiArt-Style 모델 로드 완료! (137개 예술 스타일 지원)")

def get_art_classes():
    """예술 분류 클래스 반환"""
    if art_model:
        return art_model.config.id2label
    return {}

art_classes = get_art_classes()

def classify_with_art_model(image_tensor):
    """WikiArt-Style 모델로 예술 분류"""
    try:
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)
        
        image_pil = transforms.ToPILImage()(image_tensor.clamp(0, 1))
        inputs = art_processor(images=image_pil, return_tensors="pt")
        
        with torch.no_grad():
            outputs = art_model(**inputs)
            predictions = F.softmax(outputs.logits, dim=-1)
            
        predicted_idx = predictions.argmax()  
        confidence = predictions.max()       
        predicted_class = art_classes.get(predicted_idx.item(), "Unknown_Style")
        
        return predicted_class, float(confidence), int(predicted_idx)
    
    except Exception as e:
        print(f"❌ 예술 분류 실패: {e}")
        return "Post-Impressionism", 0.75, 0

def flexible_resize_transform(image, max_size=224):
    original_size = image.size
    
    if original_size[0] > original_size[1]:
        new_width = max_size
        new_height = int(max_size * original_size[1] / original_size[0])
    else:
        new_height = max_size
        new_width = int(max_size * original_size[0] / original_size[1])
    
    resized_image = image.resize((new_width, new_height), Image.BICUBIC)
    
    padding_left = (max_size - new_width) // 2
    padding_top = (max_size - new_height) // 2
    padding_right = max_size - new_width - padding_left
    padding_bottom = max_size - new_height - padding_top
    
    padded_image = ImageOps.expand(
        resized_image, 
        border=(padding_left, padding_top, padding_right, padding_bottom), 
        fill=0
    )
    
    img_np = np.array(padded_image).astype(np.float32) / 255.0
    img_np = img_np.transpose((2, 0, 1))
    
    return torch.from_numpy(img_np)

def fgsm_attack_with_blur(image_tensor, base_epsilon=0.015, base_sigma=0.4):
    image_tensor = image_tensor.clone().unsqueeze(0).requires_grad_(True)
    
    original_class, conf, original_pred = classify_with_art_model(image_tensor)
    
    # 적응적 엡실론 조정 
    if conf > 0.99:
        eps = base_epsilon * 4.0
        sigma = base_sigma * 0.3
    elif conf > 0.95:
        eps = base_epsilon * 2.5
        sigma = base_sigma * 0.5
    elif conf > 0.9:
        eps = base_epsilon * 1.5
        sigma = base_sigma
    else:
        eps = base_epsilon
        sigma = base_sigma
    
    # FGSM 공격
    try:
        # gradient 계산
        image_pil = transforms.ToPILImage()(image_tensor.squeeze().clamp(0, 1))
        inputs = art_processor(images=image_pil, return_tensors="pt")
        inputs['pixel_values'].requires_grad_(True)
        
        outputs = art_model(**inputs)
        target = torch.tensor([original_pred])
        loss = F.cross_entropy(outputs.logits, target)
        
        # Gradient 기반 perturbation
        loss.backward()
        
        if inputs['pixel_values'].grad is not None:
            perturbation = eps * inputs['pixel_values'].grad.sign()
            # 크기 맞춤
            if perturbation.shape != image_tensor.shape:
                perturbation = F.interpolate(perturbation, size=image_tensor.shape[2:], mode='bilinear')
            adv_image = image_tensor + perturbation
            print(f"gradient 기반 FGSM 적용!")
        else:
            raise Exception("Gradient 계산 실패")
            
    except Exception as e:
        print(f"⚠️ 예술 모델 gradient 실패, fallback 사용: {e}")
        # 기존 방식으로 fallback
        perturbation = eps * torch.randn_like(image_tensor)
        adv_image = image_tensor + perturbation
    
    adv_image = torch.clamp(adv_image, 0, 1)
    
    # 가우시안 블러
    adv_np = adv_image.squeeze(0).detach().cpu().numpy()
    adv_blur_np = np.stack([gaussian_filter(c, sigma=sigma) for c in adv_np])
    adv_blur = torch.from_numpy(adv_blur_np).unsqueeze(0)

    # 적대적 예술 분류
    adversarial_class, adversarial_conf, adversarial_pred = classify_with_art_model(adv_blur)
    
    attack_success = original_pred != adversarial_pred
    confidence_drop = conf - adversarial_conf
    
    # 로그 출력
    print(f"🎨 원본: {original_class} ({conf:.3f})")
    print(f"🎯 공격후: {adversarial_class} ({adversarial_conf:.3f})")
    print(f"📊 성공: {attack_success}, 신뢰도 변화: {confidence_drop:.3f}")

    return {
        'original_class': original_class,
        'adversarial_class': adversarial_class,
        'original_conf': float(conf),
        'adversarial_conf': float(adversarial_conf),
        'attack_success': bool(attack_success),
        'confidence_drop': float(confidence_drop),
        'epsilon_used': float(eps),
        'sigma_used': float(sigma),
        'original_image': image_tensor.squeeze(0).detach(),
        'adversarial_image': adv_blur.squeeze(0).detach()
    }

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def tensor_to_base64(tensor):
    try:
        # 텐서를 numpy로 변환
        img_np = tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        # PIL 이미지로 변환
        img = Image.fromarray(img_np)
        
        # 메모리 버퍼에 PNG로 저장
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        
        # Base64 인코딩
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 표준 Data URL 형식으로 반환 (S3 URL 아님)
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"❌ Base64 변환 실패: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 및 적대적 노이즈 생성"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'}), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': '유효하지 않은 파일입니다'}), 400
        
        # 이미지 처리
        img = Image.open(file.stream).convert('RGB')
        img_tensor = flexible_resize_transform(img)
        
        # FGSM 적대적 공격 수행
        result = fgsm_attack_with_blur(img_tensor)
        
        # Base64 변환
        original_base64 = tensor_to_base64(result['original_image'])
        processed_base64 = tensor_to_base64(result['adversarial_image'])
        
        if not original_base64 or not processed_base64:
            return jsonify({'error': 'Base64 변환 실패'}), 500
        
        # ✅ 백엔드 NoiseFlaskResponseDTO와 정확히 일치하는 필드명 사용
        return jsonify({
            'originalFilePath': original_base64,           
            'processedFilePath': processed_base64,         
            'epsilon': float(result['epsilon_used']),      
            'attackSuccess': bool(result['attack_success']), 
            'originalPrediction': str(result['original_class']), 
            'adversarialPrediction': str(result['adversarial_class']), 
            'originalConfidence': f"{result['original_conf']:.3f}", 
            'adversarialConfidence': f"{result['adversarial_conf']:.3f}", 
            'confidenceDrop': f"{result['confidence_drop']*100:.1f}%", 
            'message': '적대적 노이즈 생성 완료'          
        })
        
    except Exception as e:
        print(f"❌ Flask 오류: {e}")
        return jsonify({'error': f'처리 중 오류: {str(e)}'}), 500


@app.route('/test-art-model', methods=['GET'])
def test_art_model():
    try:
        if not art_model:
            return jsonify({
                'error': '예술 분류 모델이 로드되지 않았습니다',
                'modelStatus': '실패',
                'supportedClasses': 0
            }), 500
        
        return jsonify({
            'modelStatus': '정상',
            'modelName': 'WikiArt-Style (137 Classes)',
            'supportedClasses': len(art_model.config.id2label),
            'sampleClasses': list(art_model.config.id2label.values())[:15],
            'message': '🎨 WikiArt-Style 예술 분류 모델 정상 동작'
        })
        
    except Exception as e:
        return jsonify({'error': f'모델 테스트 실패: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)