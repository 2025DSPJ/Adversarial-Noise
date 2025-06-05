from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage import gaussian_filter
import base64
import io
import os
import requests
import json

app = Flask(__name__)

# 모델 로드
print("모델 로딩 중...")
model = models.resnet101(pretrained=True)
model.eval()

# ImageNet 클래스 로드
def load_imagenet_classes():
    try:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url, timeout=10)
        classes = response.text.strip().split('\n')
        return {str(i): classes[i] for i in range(len(classes))}
    except:
        return {'151': 'Chihuahua', '235': 'German shepherd'}

imagenet_classes = load_imagenet_classes()

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

def fgsm_attack_with_blur(image_tensor, model, base_epsilon=0.015, base_sigma=0.4):
    image_tensor = image_tensor.clone().unsqueeze(0).requires_grad_(True)
    
    # 원본 예측
    with torch.no_grad():
        output = model(image_tensor)
        conf, pred = F.softmax(output, dim=1).max(1)
    
    original_class = imagenet_classes.get(str(pred.item()), f"Class_{pred.item()}")
    
    # 적응적 엡실론 조정 
    if conf.item() > 0.99:
        eps = base_epsilon * 4.0
        sigma = base_sigma * 0.3
    elif conf.item() > 0.95:
        eps = base_epsilon * 2.5
        sigma = base_sigma * 0.5
    elif conf.item() > 0.9:
        eps = base_epsilon * 1.5
        sigma = base_sigma
    else:
        eps = base_epsilon
        sigma = base_sigma
    
    # FGSM 공격
    output = model(image_tensor)
    loss = F.cross_entropy(output, pred)
    model.zero_grad()
    loss.backward()
    
    perturbation = eps * image_tensor.grad.sign()
    adv_image = image_tensor + perturbation
    adv_image = torch.clamp(adv_image, 0, 1)
    
    # 가우시안 블러
    adv_np = adv_image.squeeze(0).detach().cpu().numpy()
    adv_blur_np = np.stack([gaussian_filter(c, sigma=sigma) for c in adv_np])
    adv_blur = torch.from_numpy(adv_blur_np).unsqueeze(0)
    
    # 적대적 예측
    with torch.no_grad():
        adv_output = model(adv_blur)
        adv_conf, adv_pred = F.softmax(adv_output, dim=1).max(1)
    
    adversarial_class = imagenet_classes.get(str(adv_pred.item()), f"Class_{adv_pred.item()}")
    attack_success = pred.item() != adv_pred.item()
    confidence_drop = conf.item() - adv_conf.item()
    
    return {
        'original_class': original_class,
        'adversarial_class': adversarial_class,
        'original_conf': conf.item(),
        'adversarial_conf': adv_conf.item(),
        'attack_success': attack_success,
        'confidence_drop': confidence_drop,
        'epsilon_used': eps,
        'sigma_used': sigma,
        'original_image': image_tensor.squeeze(0).detach(),
        'adversarial_image': adv_blur.squeeze(0).detach()
    }

# 파일 확장자 검사 함수 
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def tensor_to_base64(tensor):
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Flask 라우트들
@app.route('/')
def index():
    return render_template('index.html')

# 업로드 라우트 수정
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일을 선택해주세요'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': '지원하지 않는 파일 형식입니다'})
        
        # 이미지 처리
        img = Image.open(file.stream).convert('RGB')
        img_tensor = flexible_resize_transform(img)
        result = fgsm_attack_with_blur(img_tensor, model)
        
        return jsonify({
            'originalFilePath': tensor_to_base64(result['original_image']),
            'processedFilePath': tensor_to_base64(result['adversarial_image']),
            'epsilon': float(result.get('epsilon_used', 0.03)),
            'attackSuccess': result.get('attack_success', False),
            'originalPrediction': result.get('original_class', 'Unknown'),
            'adversarialPrediction': result.get('adversarial_class', 'Unknown'),
            'originalConfidence': f"{result.get('original_conf', 0):.3f}",
            'adversarialConfidence': f"{result.get('adversarial_conf', 0):.3f}",
            'confidenceDrop': f"{result.get('confidence_drop', 0)*100:.1f}%",
            'message': '처리 완료'
        })
        
    except Exception as e:
        return jsonify({'error': f'처리 중 오류: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
