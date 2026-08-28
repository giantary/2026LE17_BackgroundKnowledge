# PyTorch Classification - Tóm tắt

## Tổng quan

Phân loại (classification) là bài toán dự đoán một mẫu dữ liệu thuộc lớp nào.

- **Phân loại nhị phân (Binary)**: 2 lớp (ví dụ: spam / không spam)
- **Phân loại đa lớp (Multi-class)**: 3 lớp trở lên (ví dụ: chó / mèo / gà)

---

## 0. Kiến trúc mô hình phân loại

| Siêu tham số | Phân loại nhị phân | Phân loại đa lớp |
|---|---|---|
| Đầu vào (Input) | Số lượng đặc trưng | Giống nhị phân |
| Lớp ẩn (Hidden layers) | Tối thiểu 1, thường 10-512 nơ-ron | Giống nhị phân |
| Đầu ra (Output) | 1 giá trị | 1 giá trị cho mỗi lớp |
| Hàm kích hoạt ẩn | ReLU | Giống nhị phân |
| Hàm kích hoạt đầu ra | Sigmoid | Softmax |
| Hàm mất mát | `nn.BCEWithLogitsLoss` | `nn.CrossEntropyLoss` |
| Optimizer | SGD, Adam | Giống nhị phân |

---

## 1. Tạo dữ liệu

### Dữ liệu nhị phân (hình tròn)

```python
from sklearn.datasets import make_circles
X, y = make_circles(1000, noise=0.03, random_state=42)
```

### Chuyển sang tensor và chia train/test

```python
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 2. Xây dựng mô hình

### Mô hình cơ bản (chỉ tuyến tính)

```python
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
```

### Mô hình phi tuyến tính (có ReLU)

```python
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
```

---

## 3. Huấn luyện mô hình

### Hàm mất mát và Optimizer cho phân loại nhị phân

```python
loss_fn = nn.BCEWithLogitsLoss()  # Tích hợp sẵn sigmoid
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

### Hàm tính accuracy

```python
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
```

### Chuyển logits thành nhãn dự đoán

Mô hình trả về **logits** (giá trị thô). Để có nhãn dự đoán:

```
logits -> sigmoid -> xác suất -> làm tròn -> nhãn
```

```python
y_pred = torch.round(torch.sigmoid(model(X_test)))
```

### Vòng lặp huấn luyện

```python
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze()
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, torch.round(torch.sigmoid(y_logits)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, torch.round(torch.sigmoid(test_logits)))
```

---

## 4-5. Cải thiện mô hình

### Vấn đề: Mô hình tuyến tính không học được dữ liệu phi tuyến

Thêm lớp và nơ-ron (nhưng vẫn tuyến tính) không giúp ích. Cần **hàm kích hoạt phi tuyến tính**.

### Tuyến tính vs Phi tuyến tính

- **Tuyến tính**: Chỉ dùng đường thẳng để mô hình hóa
- **Phi tuyến tính**: Dùng đường cong để mô hình hóa

### Các hàm kích hoạt phi tuyến phổ biến

| Tên | Công thức | PyTorch |
|---|---|---|
| Sigmoid | sigmoid(x) | `nn.Sigmoid` |
| ReLU | max(0, x) | `nn.ReLU` |
| Tanh | tanh(x) | `nn.Tanh` |

### Kết quả

Mô hình có ReLU học được ranh giới phi tuyến tính, phân loại chính xác dữ liệu hình tròn.

---

## 8. Phân loại đa lớp

### Tạo dữ liệu

```python
from sklearn.datasets import make_blobs
X_blob, y_blob = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=42)
```

### Mô hình đa lớp

```python
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
```

### Hàm mất mát và Optimizer cho đa lớp

```python
loss_fn = nn.CrossEntropyLoss()  # Tích hợp sẵn softmax
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

### Chuyển logits thành nhãn (đa lớp)

```
logits -> softmax -> xác suất -> argmax -> nhãn
```

```python
y_pred = torch.softmax(model(X_test), dim=1).argmax(dim=1)
```

---

## 9. Các chỉ số đánh giá phân loại

| Chỉ số | Định nghĩa |
|---|---|
| **Accuracy** | Tỉ lệ dự đoán đúng trên tổng số mẫu |
| **Precision** | Tỉ lệ true positive / tổng số dự đoán dương. Precision cao = ít false positive |
| **Recall** | Tỉ lệ true positive / (true positive + false negative). Recall cao = ít false negative |
| **F1-score** | Kết hợp precision và recall. 1 = tốt nhất, 0 = tệ nhất |
| **Confusion matrix** | Bảng so sánh dự đoán với nhãn thực |
| **Classification report** | Tập hợp precision, recall, f1-score |

Thư viện: `torchmetrics` hoặc `sklearn.metrics`

```python
from torchmetrics import Accuracy
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)
torchmetrics_accuracy(y_preds, y_blob_test)
```

---

## Tóm tắt quy trình phân loại

### Bước 1: Chuẩn bị dữ liệu
- Tạo/tải dữ liệu, chuyển thành tensor
- Chia train/test

### Bước 2: Xây dựng mô hình
- Kế thừa `nn.Module`
- Thêm hàm kích hoạt phi tuyến (ReLU) giữa các lớp

### Bước 3: Chọn hàm mất mát và optimizer
- Nhị phân: `nn.BCEWithLogitsLoss()` + SGD/Adam
- Đa lớp: `nn.CrossEntropyLoss()` + SGD/Adam

### Bước 4: Huấn luyện
- forward -> loss -> zero_grad -> backward -> step
- Theo dõi loss và accuracy

### Bước 5: Đánh giá
- Dùng `torch.inference_mode()`
- Chuyển logits thành nhãn (sigmoid+round hoặc softmax+argmax)
- Trực quan hóa ranh giới quyết định

### Điểm quan trọng
- Mô hình chỉ có lớp tuyến tính không thể học mẫu phi tuyến
- Thêm hàm kích hoạt phi tuyến (ReLU, Sigmoid, Tanh) để mô hình linh hoạt hơn
- `BCEWithLogitsLoss` đã tích hợp sigmoid, không cần thêm sigmoid ở output
- `CrossEntropyLoss` đã tích hợp softmax, không cần thêm softmax ở output
