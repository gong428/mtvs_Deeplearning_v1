import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# 전역 변수
num_epochs = 50
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
test_loss = 0.0
test_accuracy = 0.0

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch)

def calculate_accuracy(outputs, labels):
    preds = (outputs > 0.5).float()
    correct = ((preds * labels).sum(dim=1) > 0).float()
    accuracy = correct.mean().item()
    return accuracy

def train_and_validate_best(model, train_loader, val_loader, optimizer, criterion, epochs, device, class_num):
    global train_losses, val_losses, train_accuracies, val_accuracies
    best_val_accuracy = 0.0
    min_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        total_train = 0
        print("학습 시작")

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 정확도 계산
            train_accuracy = calculate_accuracy(outputs, labels)
            train_correct += train_accuracy * inputs.size(0)
            total_train += inputs.size(0)

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracy = train_correct / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                # 정확도 계산
                val_accuracy = calculate_accuracy(outputs, labels)
                val_correct += val_accuracy * inputs.size(0)
                total_val += inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracy = val_correct / total_val
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}] - Training loss: {epoch_loss:.3f}, Validation loss: {epoch_val_loss:.3f}, Train accuracy: {train_accuracy:.2%}, Val accuracy: {val_accuracy:.2%}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'val_acc_best_model.pth')
            print(f'Best model based on validation accuracy saved at epoch {epoch + 1}')

        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'val_loss_best_model.pth')
            print(f'Best model based on validation loss saved at epoch {epoch + 1}')

    # CSV 파일로 저장
    metrics_df = pd.DataFrame({
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    })
    metrics_df.to_csv('training_metrics.csv', index=False)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 정확도 계산
            accuracy = calculate_accuracy(outputs, labels)
            correct += accuracy * inputs.size(0)
            total += inputs.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def balance_labels(df, sample_size):
    balanced_dfs = []
    image_paths = df['Image Path']
    label_columns = df.columns[:-1]
    
    for label in label_columns:
        label_df = df[df[label] == 1]
        sampled_label_df = label_df.sample(n=min(len(label_df), sample_size), random_state=42)
        balanced_dfs.append(sampled_label_df)
    
    final_df = pd.concat(balanced_dfs).reset_index(drop=True)
    final_df['Image Path'] = image_paths.loc[final_df.index].values
    
    return final_df

def create_image_paths(df, column_name, base_path='../cut_datas'):
    df['Image Path'] = df[column_name].apply(lambda x: os.path.join(base_path, f'{x}.jpg'))
    return df


class ClothesDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.image_paths = df['Image Path'].apply(lambda x: os.path.abspath(os.path.join('../data', x))).to_numpy()
        self.labels = np.array(df.iloc[:, :-1])
        print(f'Labels shape: {self.labels.shape}')
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None, img_path

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image.clone().detach().float(), torch.tensor(label, dtype=torch.float32)



def main():
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터프레임 로드
    df_raw = pd.read_csv('multi_label_dataset_cut.csv')
    df = df_raw.copy()
    df = df.drop(['Image Filename', 'None'], axis=1)

    # 이미지 경로 생성
    df = create_image_paths(df, 'Image ID')
    df = df.drop(['Unnamed: 0', 'Image ID'], axis=1)
    df = df.drop(['Unnamed: 0.1'], axis=1)
    #물흐리는 스트리트 삭제
    df = df.drop(['스트리트'], axis=1)

    # 라벨 균형 맞추기
    df = balance_labels(df, sample_size=1000)
    label_columns = df.columns[:-1]

    # 데이터셋 및 데이터 로더 설정
    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(saturation=0.3,brightness=0.3,),  # 채도를 0.5 배까지 무작위로 변경
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

    # 검증 및 테스트용 변환 설정
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = ClothesDataset(df=df, transform=train_transform)
    val_test_dataset = ClothesDataset(df=df, transform=val_test_transform)

    # 전체 데이터셋의 인덱스를 생성
    indices = list(range(len(train_dataset)))
    train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=2/9, random_state=42)

    # Subset으로 나누기
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_test_dataset, val_indices)
    test_dataset = Subset(val_test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True, collate_fn=collate_fn)

    # 모델 생성 및 전이학습
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),  # Batch Normalization 추가
    nn.Dropout(0.4),  # Dropout 확률 증가
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),  # Batch Normalization 추가
    nn.Dropout(0.4),  # Dropout 확률 증가
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),  # Batch Normalization 추가
    nn.Dropout(0.4),  # Dropout 확률 증가
    nn.Linear(256, len(label_columns))  # len(label_columns)는 클래스 수
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 학습 및 검증 수행
    train_and_validate_best(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, len(label_columns))

    # 테스트 수행
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy * 100:.2f}%')


    # CSV 파일에서 데이터를 읽고 그래프 생성
    metrics_df = pd.read_csv('training_metrics.csv')

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df['train_loss'], label='Train Loss')
    plt.plot(metrics_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(metrics_df['train_accuracy'], label='Train Accuracy')
    plt.plot(metrics_df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # 테스트 결과 저장
    with open('test_results.txt', 'w') as f:
        f.write(f'Test Loss: {test_loss:.3f}\n')
        f.write(f'Test Accuracy: {test_accuracy * 100:.2f}%\n')

if __name__ == "__main__":
    main()