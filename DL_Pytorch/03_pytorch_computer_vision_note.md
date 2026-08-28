# PyTorch Computer Vision - Tóm tắt

## Tổng quan

Áp dụng quy trình PyTorch vào bài toán thị giác máy tính (Computer Vision), cụ thể là phân loại ảnh đa lớp với tập dữ liệu FashionMNIST (10 loại quần áo, ảnh xám 28x28).

---

## 0. Các thư viện Computer Vision trong PyTorch

| Thư viện | Chức năng |
|---|---|
| `torchvision` | Chứa datasets, model architectures, image transforms |
| `torchvision.datasets` | Các tập dữ liệu mẫu (FashionMNIST, CIFAR10...) |
| `torchvision.models` | Các kiến trúc mô hình CV hiệu suất cao |
| `torchvision.transforms` | Các phép biến đổi hình ảnh |
| `torch.utils.data.DataLoader` | Tạo mini-batches từ dataset |

---

## 1. Tải dữ liệu

### FashionMNIST

- 60,000 ảnh huấn luyện + 10,000 ảnh kiểm tra
- 10 lớp quần áo (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- Kích thước: `[1, 28, 28]` (1 kênh màu, 28x28 pixel)

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)
```

### Trực quan hóa dữ liệu

```python
image, label = train_data[0]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(train_data.classes[label])
```

---

## 2. Chuẩn bị dữ liệu với DataLoader

DataLoader chuyển dataset thành các mini-batches, giúp:
- Tiết kiệm bộ nhớ (không tải toàn bộ dữ liệu cùng lúc)
- Cập nhật gradient hiệu quả hơn

```python
from torch.utils.data import DataLoader

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
```

---

## 3. Mô hình 0: Baseline (chỉ Linear)

Mô hình đơn giản nhất, dùng `nn.Flatten()` để trải phẳng ảnh 28x28 thành vector 784.

```python
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

model_0 = FashionMNISTModelV0(
    input_shape=784,    # 28*28
    hidden_units=10,
    output_shape=len(train_data.classes)  # 10 lớp
)
```

### Hàm mất mát và Optimizer

```python
loss_fn = nn.CrossEntropyLoss()  # Phân loại đa lớp
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)
```

### Vòng lặp huấn luyện với DataLoader

```python
for epoch in range(epochs):
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))
```

---

## 4. Đo thời gian huấn luyện

```python
from timeit import default_timer as timer

train_time_start = timer()
# ... huấn luyện ...
train_time_end = timer()
total_time = train_time_end - train_time_start
```

Dùng để so sánh hiệu suất giữa các mô hình và giữa CPU/GPU.

---

## 5. Device-agnostic code

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Trong vòng lặp, chuyển dữ liệu sang device
X, y = X.to(device), y.to(device)
```

---

## 6. Mô hình 1: Thêm ReLU (phi tuyến tính)

```python
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer_stack(x)
```

Thêm ReLU giữa các lớp Linear giúp mô hình học được các mẫu phi tuyến.

---

## 7. Mô hình 2: CNN (Convolutional Neural Network)

Trọng tâm của Computer Vision hiện đại. CNN gồm 3 thành phần chính:

### Các lớp CNN cốt lõi

| Lớp | Chức năng | PyTorch |
|---|---|---|
| **Convolution** | Trích xuất đặc trưng từ ảnh (cạnh, góc, kết cấu...) | `nn.Conv2d()` |
| **Activation** | Thêm tính phi tuyến | `nn.ReLU()` |
| **Pooling** | Giảm kích thước không gian, giữ thông tin quan trọng | `nn.MaxPool2d()` |

### Siêu tham số của Conv2d

| Tham số | Ý nghĩa |
|---|---|
| `in_channels` | Số kênh đầu vào (1 cho ảnh xám, 3 cho ảnh màu) |
| `out_channels` | Số bộ lọc (filters), cũng là số kênh đầu ra |
| `kernel_size` | Kích thước bộ lọc (thường 3x3) |
| `stride` | Bước nhảy của bộ lọc (mặc định 1) |
| `padding` | Thêm pixel xung quanh ảnh (thường 1 cho kernel 3x3) |

### Kiến trúc TinyVGG

```python
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
```

Luồng dữ liệu: `[batch, 1, 28, 28]` -> Conv Block 1 -> `[batch, hidden, 14, 14]` -> Conv Block 2 -> `[batch, hidden, 7, 7]` -> Flatten -> Linear -> `[batch, 10]`

---

## 8. So sánh các mô hình

| Mô hình | Kiến trúc | Thiết bị | Test Accuracy |
|---|---|---|---|
| Model 0 | Flatten + Linear | CPU | ~83% |
| Model 1 | Flatten + Linear + ReLU | GPU/CPU | ~83% |
| Model 2 | CNN (TinyVGG) | GPU | ~88-90% (tốt nhất) |

CNN vượt trội vì tận dụng cấu trúc không gian của ảnh thông qua convolution.

---

## 9-10. Đánh giá mô hình tốt nhất

### Dự đoán và trực quan hóa

```python
model.eval()
with torch.inference_mode():
    y_pred = model(image.unsqueeze(0).to(device))
    y_pred_label = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
```

### Ma trận nhầm lẫn (Confusion Matrix)

Dùng `torchmetrics.ConfusionMatrix` và `mlxtend.plotting.plot_confusion_matrix` để xem mô hình hay nhầm lớp nào.

```python
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor, target=y_test_tensor)
plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), class_names=class_names)
```

---

## 11. Lưu và tải mô hình

```python
# Lưu
torch.save(model_2.state_dict(), "03_pytorch_computer_vision_model_2.pth")

# Tải
loaded_model = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=10)
loaded_model.load_state_dict(torch.load("03_pytorch_computer_vision_model_2.pth"))
```

---

## Tóm tắt quy trình Computer Vision

### Bước 1: Tải dữ liệu
- Dùng `torchvision.datasets` để tải dataset
- Áp dụng `transforms.ToTensor()` để chuyển ảnh thành tensor

### Bước 2: Tạo DataLoader
- Chia thành mini-batches với `DataLoader`
- `shuffle=True` cho train, `shuffle=False` cho test

### Bước 3: Xây dựng mô hình
- Bắt đầu với baseline đơn giản (Linear)
- Thêm ReLU để tăng khả năng phi tuyến
- Dùng CNN (Conv2d + ReLU + MaxPool2d) cho kết quả tốt nhất

### Bước 4: Huấn luyện
- `CrossEntropyLoss` cho phân loại đa lớp
- Lặp qua các batches trong DataLoader
- Đo thời gian để so sánh CPU vs GPU

### Bước 5: Đánh giá
- So sánh loss và accuracy giữa các mô hình
- Dùng Confusion Matrix để phân tích lỗi
- Trực quan hóa dự đoán

### Bước 6: Lưu mô hình
- `torch.save(model.state_dict(), path)`

### Điểm quan trọng
- Luôn bắt đầu với baseline đơn giản trước
- CNN trích xuất đặc trưng không gian tốt hơn Linear
- `nn.Flatten()` cần thiết khi chuyển từ Conv2d sang Linear
- Kích thước đầu vào của lớp Linear cuối phụ thuộc vào đầu ra của Conv blocks
- Dùng `torchmetrics` và confusion matrix để đánh giá chi tiết
