import os
import random
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

def load_and_preprocess_image(image_path, transform):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def predict_and_display_images(model, image_folder, transform, device, num_images=30):
    # 모델을 평가 모드로 설정
    model.eval()

    # 이미지 폴더에서 파일 목록 가져오기
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    # 랜덤하게 선택된 이미지 파일 가져오기
    selected_images = image_files[:num_images]

    # 이미지와 예측 결과를 저장할 리스트
    images = []
    predictions = []

    for image_path in selected_images:
        image = load_and_preprocess_image(image_path, transform)
        if image is not None:
            images.append(image)
            image = image.unsqueeze(0).to(device)  # 배치 차원 추가 및 디바이스로 이동
            with torch.no_grad():
                outputs = model(image)
                outputs = torch.sigmoid(outputs)
            predictions.append(outputs.cpu().numpy().flatten())

    # 이미지와 예측 결과 출력
    fig, axs = plt.subplots(num_images // 5, 5, figsize=(20, num_images // 5 * 4))
    for i, (img, pred) in enumerate(zip(images, predictions)):
        ax = axs[i // 5, i % 5]
        img = transforms.ToPILImage()(img.cpu())
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Pred: {pred}')

    plt.tight_layout()
    plt.show()

def main():
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 생성 및 전이학습
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 11)  # 클래스 수는 실제 데이터셋에 맞게 조정하세요
    )

    model = model.to(device)

    # 학습된 모델 로드
    model.load_state_dict(torch.load('best_model_integrated_v7.pth'))
    
    # 이미지 변환 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 임의의 이미지를 예측하고 출력
    image_folder = '../test/로맨틱'  # 이미지가 저장된 폴더 경로
    predict_and_display_images(model, image_folder, transform, device)

if __name__ == "__main__":
    main()
