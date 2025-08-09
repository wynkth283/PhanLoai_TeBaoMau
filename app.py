import matplotlib
matplotlib.use('Agg')  # Sử dụng backend Agg không tương tác
from flask import Flask, request, render_template, jsonify, url_for
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import io
import base64
import time

app = Flask(__name__)

# Đảm bảo các thư mục tồn tại
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/gradcam_results', exist_ok=True)  # Đảm bảo thư mục static/gradcam_results tồn tại

# Cấu hình
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = 'best_vit_blood_cell_model_fuzzy.pth'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Định nghĩa lớp
classes = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'platelet']
label_to_idx = {cls: i for i, cls in enumerate(classes)}
num_classes = len(classes)

# Tiền xử lý dựa trên logic mờ (Fuzzy Logic)
def preprocess_and_segment_nucleus_fuzzy(image):
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    intensity = enhanced.flatten()
    intensity_range = np.linspace(0, 255, 100)
    low_intensity = fuzz.trimf(intensity_range, [0, 0, 100])
    medium_intensity = fuzz.trimf(intensity_range, [50, 128, 200])
    high_intensity = fuzz.trimf(intensity_range, [150, 255, 255])
    low_mem = fuzz.interp_membership(intensity_range, low_intensity, intensity)
    medium_mem = fuzz.interp_membership(intensity_range, medium_intensity, intensity)
    high_mem = fuzz.interp_membership(intensity_range, high_intensity, intensity)
    nucleus_mem = high_mem.reshape(enhanced.shape)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].flatten()
    saturation = hsv[:, :, 1].flatten()
    hue_range = np.linspace(0, 180, 100)
    sat_range = np.linspace(0, 255, 100)
    purple_hue = fuzz.trimf(hue_range, [110, 140, 170])
    orange_hue = fuzz.trimf(hue_range, [0, 15, 30])
    high_sat = fuzz.trimf(sat_range, [60, 128, 255])
    purple_mem = fuzz.interp_membership(hue_range, purple_hue, hue) * fuzz.interp_membership(sat_range, high_sat, saturation)
    orange_mem = fuzz.interp_membership(hue_range, orange_hue, hue) * fuzz.interp_membership(sat_range, high_sat, saturation)
    granule_mem = np.maximum(purple_mem, orange_mem).reshape(enhanced.shape)
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    sigma = min(h, w) / 4
    y, x = np.ogrid[:h, :w]
    gaussian_mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    gaussian_mask = gaussian_mask / gaussian_mask.max()
    attention_mask = 0.5 * nucleus_mem + 0.2 * granule_mem + 0.3 * gaussian_mask
    attention_mask = np.clip(attention_mask, 0, 1)
    attention_mask_rgb = np.stack([attention_mask] * 3, axis=-1).astype(np.float32)
    blurred_bg = cv2.GaussianBlur(img_np, (15, 15), 0)
    masked_image = img_np.astype(np.float32) * attention_mask_rgb + blurred_bg.astype(np.float32) * (1 - attention_mask_rgb)
    masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)
    sobelx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    edge_rgb = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
    final_image = (0.7 * masked_image + 0.3 * edge_rgb).astype(np.uint8)
    return Image.fromarray(final_image)

# Biến đổi dữ liệu đầu vào
transform = transforms.Compose([
    transforms.Lambda(preprocess_and_segment_nucleus_fuzzy),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tạo mô hình ViT
def create_vit_model(num_classes):
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# Tải mô hình
device = torch.device('cpu')  # Force CPU usage
model = create_vit_model(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Đã tải mô hình huấn luyện sẵn: best_vit_blood_cell_model_fuzzy.pth")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    orig_image = image.resize((224, 224))
    orig_image_np = np.array(orig_image) / 255.0
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        pred_confidence = probs[pred_idx] * 100
    
    # Sinh ảnh Grad-CAM
    target_layers = [model.patch_embed.proj]
    cam = GradCAM(model=model, target_layers=target_layers)
    target = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=target)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(orig_image_np, grayscale_cam, use_rgb=True)
    
    # Lưu ảnh Grad-CAM với timestamp để tránh cache
    timestamp = int(time.time())
    gradcam_filename = f'gradcam_output_{timestamp}.png'
    gradcam_path = os.path.join('static', 'gradcam_results', gradcam_filename)
    plt.imsave(gradcam_path, visualization)
    print(f"Đã lưu ảnh Grad-CAM: {gradcam_path}")
    
    # Vẽ biểu đồ xác suất dự đoán
    plt.figure(figsize=(6, 4))
    plt.bar(classes, probs * 100, color='skyblue')
    plt.xlabel('Các lớp')
    plt.ylabel('Xác suất (%)')
    plt.title('Lớp dự đoán')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Lưu biểu đồ thành base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close('all')  # Đóng tất cả figure để tránh rò rỉ bộ nhớ
    
    # Trả về đường dẫn tương đối cho ảnh gradcam
    gradcam_url = f'/static/gradcam_results/{gradcam_filename}'
    
    return {
        'pred_label': pred_label,
        'pred_confidence': pred_confidence,
        'probabilities': {cls: prob * 100 for cls, prob in zip(classes, probs)},
        'gradcam_path': gradcam_url,
        'plot_base64': plot_base64
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được tải lên'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            result = process_image(filepath)
            return jsonify({
                'pred_label': result['pred_label'],
                'pred_confidence': result['pred_confidence'],
                'probabilities': result['probabilities'],
                'gradcam_image': result['gradcam_path'],
                'plot_base64': result['plot_base64']
            })
        except Exception as e:
            print(f"Lỗi xử lý ảnh: {str(e)}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Định dạng file không hợp lệ'}), 400

if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # Tắt chế độ đa luồng