import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import csv
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

n = 0
name = "Outputs-Unsaturated concentration"

pattern = list()
fhd = csv.reader(open('Inputs.csv', 'r', encoding='utf-8-sig'))
for line in fhd:
    pattern.append(line)
pattern = np.array(pattern, dtype='float64')
pattern = pattern.tolist()

label = list()
fhl = csv.reader(open('%s.csv' % name, 'r', encoding='utf-8-sig'))
for line in fhl:
    label.append(line)
label = np.array(label, dtype='float64')
label = label.tolist()


# Data augmentation and train-test split code here...
#
#
# def random_jitter(data, sigma=0.05):
#     jittered_data = data + np.random.normal(0, sigma, data.shape)
#     return jittered_data
#
#
# def add_noise(data, noise_factor=0.02):
#     noisy_data = data + noise_factor * np.random.randn(*data.shape)
#     return noisy_data
#
#
# def data_augmentation(data, labels, num_augmentations=2):
#     augmented_datas = []
#     augmented_labelss = []
#
#     for _ in range(num_augmentations):
#         augmented_data = random_jitter(np.array(data))
#         augmented_data = add_noise(augmented_data)
#         augmented_datas.append(augmented_data)
#         augmented_labelss.append(np.ravel(labels))
#
#     augmented_datas = np.vstack(augmented_datas)
#     augmented_labelss = np.hstack(augmented_labelss)
#
#     augmented_datas = np.vstack((data, augmented_datas))
#     augmented_labelss = np.hstack((np.ravel(labels), augmented_labelss))
#
#     return augmented_datas, augmented_labelss


def s(pre, label):
    l = []
    error_rates = np.abs((pre - label) / label) * 100
    for i, error_rate in enumerate(error_rates):
        l.append(f'{error_rate:.2f}%')
    return l


param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4]
}

model_rf = RandomForestRegressor(max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=100,
                                 random_state=1)


# 划分训练集和临时集 (包括验证集和测试集)
temp_data, test_data, temp_labels, test_labels = train_test_split(pattern, label, test_size=0.21, random_state=42)
# 这行代码将原始数据集（pattern 和 label）分割为两个部分：一个临时数据集（temp_data 和 temp_labels）和一个测试数据集（test_data 和 test_labels）。
# 其中，测试集的大小是原始数据的17%，即 test_size=0.17。random_state=42 确保每次运行这段代码时，数据的划分方式都是相同的。
# 进一步划分临时集为训练集和验证集
train_data, val_data, train_labels, val_labels = train_test_split(temp_data, temp_labels, test_size=0.2,
                                                                  random_state=42)
# 这行代码进一步将临时数据集分割为训练集（train_data 和 train_labels）和验证集（val_data 和 val_labels）。
# 这次，验证集的大小是临时数据的20%，即 test_size=0.2。同样，random_state=42 确保了划分的确定性。

print("训练集样本数:", len(train_data))
print("验证集样本数:", len(val_data))
print("测试集样本数:", len(test_data))
augmented_feature_datas = train_data
augmented_labelss = train_labels

print(len(augmented_feature_datas), len(augmented_labelss))

import numpy as np

# # 将数据和标签转换为NumPy数组
data = np.array(augmented_feature_datas)
labels = np.array(augmented_labelss)
# 创建 MinMaxScaler 对象
scaler = MinMaxScaler()

# 在训练集上拟合归一化器并进行归一化
scaler.fit(data)
normalized_data = scaler.transform(data)
## 对某一列进行*1.3/0.7操作
normalized_cross_data = scaler.transform(val_data)
normalized_test_data = scaler.transform(test_data)

model_rf.fit(normalized_data, augmented_labelss)

label_predict_rf = model_rf.predict(normalized_cross_data)
label_predict_rf1 = model_rf.predict(normalized_test_data)

df = pd.DataFrame()
df1 = pd.DataFrame()
df['真实值'] = [i[0] for i in val_labels]
df1['真实值'] = [i[0] for i in test_labels]

df["rf_预测"] = label_predict_rf
df1["rf_预测"] = label_predict_rf1
l3 = s(df['rf_预测'], df['真实值'])
l4 = s(df1['rf_预测'], df1['真实值'])
df["rf_误差率"] = l3
df1["rf_误差率"] = l4

df.to_csv("RF-Outputs-Unsaturated concentration-test.csv")
df1.to_csv("RF-Outputs-Unsaturated concentration-val.csv")
print(df)
print(df1)
