import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Learner:
    def __init__(
        self,
        dataset_path,
        model_name,
        batch_size=32,
        epochs=10,
        lr=0.001,
        work_dir="checkpoints",
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.work_dir = work_dir
        self.device = torch.device("cpu")

        # Load dataset
        self.train_loader, self.val_loader, self.class_names = (
            self.load_data()
        )  # ants and bees

        # Load model
        self.model = self.load_model(model_name)

        # Khởi tạo Optimizer & Scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Tạo thư mục lưu best model
        os.makedirs(self.work_dir, exist_ok=True)
        self.best_acc = 0  # Theo dõi best accuracy
        self.patience = 5  # Early stopping nếu val loss không giảm trong 5 epochs
        self.counter = 0  # Đếm số epochs không cải thiện

    def load_data(self):
        transform_train = transforms.Compose(
            [  # tăng cường dữ liệu bằng các phép biến đổi
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor(),  # chuyển đổi ảnh sang số
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # chuẩn hóa
            ]
        )

        transform_val = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, "train"), transform=transform_train
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, "val"), transform=transform_val
        )

        # Load toàn bộ dataset vào RAM để tăng tốc xử lý
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        print(
            f"Dataset Loaded: {len(train_dataset)} train images | {len(val_dataset)} val images"
        )
        return train_loader, val_loader, train_dataset.classes

    def load_model(self, model_name):
        if model_name == "resnet":
            model = models.resnet18(pretrained=True)
            # bài toán image thường có nhiều ngõ ra (phân loại lớn) nhưng mình muốn lấy 2 thôi nên dùng fc
            model.fc = nn.Linear(
                model.fc.in_features, len(self.class_names)
            )  # 2 lớp: ong & kiến
        elif model_name == "vgg16":  # vgg16 bị underfiting
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(4096, len(self.class_names))
        else:
            raise ValueError("Chỉ hỗ trợ resnet hoặc vgg16!")

        return model.to(self.device)

    def train(self):
        print(f"Training {self.epochs} epochs on CPU...")

        for epoch in range(self.epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()  # tạo mới gradient
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()  # update weight

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(self.train_loader)
            train_acc = 100.0 * correct / total
            print(
                f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
            )

            # Đánh giá trên tập val
            if self.evaluate(epoch):
                break  # Dừng nếu Early Stopping

            self.scheduler.step(train_loss)  # Giảm learning rate

        print("Training hoàn thành!")

    def evaluate(self, epoch):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100.0 * correct / total
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")

        # Lưu best model
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            torch.save(self.model.state_dict(), f"{self.work_dir}/best_model.pth")
            print(f"Best model saved at epoch {epoch+1} with accuracy {val_acc:.2f}%")
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("⏳ Early stopping triggered!")
                return True  # Dừng training

    # load best model
    def load_best_model(self):
        best_model_path = f"{self.work_dir}/best_model.pth"
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            print("Best model loaded for inference!")
        else:
            print("No best model found!")

    # dự đoán
    def predict(self, image_path):
        import PIL.Image as Image

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            _, predicted = output.max(1)

        return self.class_names[predicted.item()]


learner = Learner(
    dataset_path="dataset",
    model_name="resnet",
    batch_size=32,
    epochs=10,
    lr=0.001,
    work_dir="checkpoints",
)
learner.train()

learner.load_best_model()
print(learner.predict("./img_test/ong4.jpg"))
