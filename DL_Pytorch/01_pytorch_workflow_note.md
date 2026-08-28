# PyTorch Workflow - Tóm tắt

## Tổng quan

Quy trình làm việc chuẩn trong PyTorch gồm 6 bước chính:

1. **Chuẩn bị dữ liệu** - Tạo hoặc tải dữ liệu, chia train/test
2. **Xây dựng mô hình** - Tạo lớp kế thừa `nn.Module`
3. **Huấn luyện mô hình** - Thiết lập loss function, optimizer, viết vòng lặp huấn luyện
4. **Đưa ra dự đoán (suy luận)** - Dùng `torch.inference_mode()` để dự đoán
5. **Lưu và tải mô hình** - Dùng `torch.save()` và `torch.load()`
6. **Tổng hợp** - Kết hợp tất cả các bước lại

---

## 1. Chuẩn bị dữ liệu

Học máy gồm hai phần cốt lõi:
1. Biến dữ liệu thành các con số (biểu diễn số)
2. Xây dựng mô hình để học cách biểu diễn đó

### Tạo dữ liệu

```python
weight = 0.7
bias = 0.3
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias
```

### Chia dữ liệu

| Tập dữ liệu | Mục đích | Tỉ lệ |
|---|---|---|
| **Training set** | Mô hình học từ dữ liệu này | ~60-80% |
| **Validation set** | Tinh chỉnh mô hình (tùy chọn) | ~10-20% |
| **Test set** | Đánh giá mô hình cuối cùng | ~10-20% |

```python
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
```

---

## 2. Xây dựng mô hình

### 4 module cốt lõi của PyTorch

| Module | Chức năng |
|---|---|
| `torch.nn` | Chứa tất cả các khối xây dựng mạng nơ-ron |
| `torch.nn.Parameter` | Lưu trữ các tham số (weight, bias) có thể được cập nhật qua gradient descent |
| `torch.nn.Module` | Lớp cơ sở cho mọi mạng nơ-ron. Yêu cầu phải cài đặt phương thức `forward()` |
| `torch.optim` | Chứa các thuật toán tối ưu hóa (SGD, Adam...) |

### Tạo mô hình bằng cách kế thừa nn.Module

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float),
                                    requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float),
                                 requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias  # y = mx + b
```

### Kiểm tra tham số mô hình

```python
model_0 = LinearRegressionModel()
model_0.parameters()     # Xem danh sách tham số
model_0.state_dict()     # Xem trạng thái (tên + giá trị tham số)
```

### Dự đoán trước khi huấn luyện (dùng torch.inference_mode)

```python
with torch.inference_mode():
    y_preds = model_0(X_test)
```

`torch.inference_mode()` tắt theo dõi gradient, giúp dự đoán nhanh hơn. Luôn dùng khi suy luận, không dùng khi huấn luyện.

---

## 3. Huấn luyện mô hình

### Thiết lập Loss Function và Optimizer

| Thành phần | Vai trò | Ví dụ |
|---|---|---|
| **Loss function** | Đo mức độ sai của dự đoán so với nhãn thực. Giá trị càng thấp càng tốt | `nn.L1Loss()` (MAE), `nn.MSELoss()`, `nn.BCEWithLogitsLoss()` |
| **Optimizer** | Cập nhật tham số để giảm loss | `torch.optim.SGD()`, `torch.optim.Adam()` |

```python
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
```

### Vòng lặp huấn luyện (Training Loop)

5 bước cơ bản cho mỗi epoch:

| Bước | Tên | Code |
|---|---|---|
| 1 | Lan truyền tiến (Forward pass) | `y_pred = model(X_train)` |
| 2 | Tính loss | `loss = loss_fn(y_pred, y_train)` |
| 3 | Đặt gradient về 0 | `optimizer.zero_grad()` |
| 4 | Lan truyền ngược (Backpropagation) | `loss.backward()` |
| 5 | Cập nhật tham số | `optimizer.step()` |

### Vòng lặp kiểm tra (Testing Loop)

3 bước (không có backpropagation và optimizer step):

| Bước | Tên | Code |
|---|---|---|
| 1 | Forward pass | `y_pred = model(X_test)` |
| 2 | Tính loss | `test_loss = loss_fn(y_pred, y_test)` |
| 3 | Tính metrics (tùy chọn) | Tùy chỉnh |

### Code huấn luyện hoàn chỉnh

```python
epochs = 100

for epoch in range(epochs):
    # --- Training ---
    model_0.train()

    y_pred = model_0(X_train)           # 1. Forward pass
    loss = loss_fn(y_pred, y_train)     # 2. Tính loss
    optimizer.zero_grad()                # 3. Zero grad
    loss.backward()                      # 4. Backpropagation
    optimizer.step()                     # 5. Cập nhật tham số

    # --- Testing ---
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)
```

---

## 4. Đưa ra dự đoán (Suy luận - Inference)

3 điều cần nhớ khi suy luận:

1. Đặt mô hình ở chế độ đánh giá: `model.eval()`
2. Dùng context manager: `with torch.inference_mode():`
3. Dữ liệu và mô hình phải ở cùng thiết bị (cùng GPU hoặc cùng CPU)

```python
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
```

---

## 5. Lưu và tải mô hình

### 3 phương thức chính

| Phương thức | Chức năng |
|---|---|
| `torch.save()` | Lưu mô hình (dùng `pickle`) |
| `torch.load()` | Tải mô hình đã lưu |
| `model.load_state_dict()` | Tải state_dict vào mô hình |

### Cách khuyến nghị: Lưu/tải state_dict

Lưu (chỉ lưu tham số, không lưu cấu trúc):

```python
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
```

Tải (cần tạo lại cấu trúc mô hình trước):

```python
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
```

---

## 6. Tổng hợp - Quy trình hoàn chỉnh

### Bước 1: Chuẩn bị dữ liệu
- Tạo hoặc tải dữ liệu
- Chia thành train/test sets
- Chuyển thành tensor PyTorch

### Bước 2: Xây dựng mô hình
- Kế thừa `nn.Module`
- Định nghĩa `__init__()` với các layer
- Định nghĩa `forward()` với phép tính

### Bước 3: Huấn luyện
- Chọn loss function phù hợp
- Chọn optimizer (SGD, Adam...)
- Viết vòng lặp: forward -> loss -> zero_grad -> backward -> step
- Theo dõi loss giảm qua các epoch

### Bước 4: Đánh giá
- `model.eval()` + `torch.inference_mode()`
- So sánh dự đoán với dữ liệu thực
- Trực quan hóa kết quả

### Bước 5: Lưu mô hình
- `torch.save(model.state_dict(), path)`
- Tải lại khi cần: tạo instance mới -> `load_state_dict()`

---

## Sử dụng nn.Linear thay vì tự tạo Parameter

PyTorch cung cấp `nn.Linear` để tự động tạo weight và bias:

```python
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
```

Cách này ngắn gọn hơn và là cách phổ biến trong thực tế.

---

## Thiết bị không phụ thuộc GPU (Device-agnostic code)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chuyển mô hình và dữ liệu sang device
model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
```

Luôn viết code theo cách này để mô hình chạy được trên cả GPU và CPU.
