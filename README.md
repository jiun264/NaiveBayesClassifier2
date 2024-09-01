```markdown
# Naive Bayes Classifier

## 概述
這個PROJECT實現了 Naive Bayes 分類器的程式碼。該分類器包含兩個版本，一個是考慮特徵獨立的情況，另一個則考慮了特徵之間的協方差。

## 使用方法

### 安裝相依性
確保您的 Python 環境中安裝了以下相依性：

```bash
pip install numpy pandas scipy scikit-learn
```

### 資料集
本項目使用了名為 "wine.data" 的資料集，包含葡萄酒的化學特性。您可以通過以下鏈接獲取資料集：[wine.data](連接到資料集)

### 執行範例
```python
# 導入必要的庫
import pandas as pd
from sklearn.preprocessing import StandardScaler
from your_module_name import NaiveBayesClassifier  # 請替換為你的模塊名稱

# 讀取資料集
feature_names = ['label', 'Alcohol', 'Malic acid', ...]  # 請添加完整的特徵名稱
data = pd.read_csv("wine.data", names=feature_names)
data = data.sample(frac=1)

# 劃分資料集
half = len(data) // 2
train = data.iloc[:half]
test = data.iloc[half:]

# 資料預處理
scaler = StandardScaler()
labels = train.iloc[:, 0].values
labels_test = test.iloc[:, 0].values
train = train.drop("label", axis=1)
test = test.drop("label", axis=1)
train_features = scaler.fit_transform(train.iloc[:, :])
test_features = scaler.transform(test.iloc[:, :])

# 初始化並訓練分類器
nb = NaiveBayesClassifier()
nb.fit(train_features, labels)

# 預測和評估
predictions = nb.predict(test_features)
accuracy = nb.score(test_features, labels_test)
print(f"Accuracy: {accuracy}")
```

## 高階功能

### 考慮特徵協方差
您可以使用 `fit2` 方法來訓練考慮特徵協方差的分類器，並使用 `Bayes_error` 方法計算 Bayes 錯誤率。

```python
nb.fit2(train_features, labels)
nb.Bayes_error()
```


