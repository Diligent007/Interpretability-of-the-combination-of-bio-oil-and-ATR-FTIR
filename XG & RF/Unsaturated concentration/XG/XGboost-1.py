import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

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


model_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',  # for regression problems
    # Add other hyperparameters as needed
)

temp_data, test_data, temp_labels, test_labels = train_test_split(pattern, label, test_size=0.21, random_state=42)

train_data, val_data, train_labels, val_labels = train_test_split(temp_data, temp_labels, test_size=0.2, random_state=42)

print("训练集样本数:", len(train_data))
print("验证集样本数:", len(val_data))
print("测试集样本数:", len(test_data))
augmented_feature_datas = train_data
augmented_labelss = train_labels

print(len(augmented_feature_datas), len(augmented_labelss))

data = np.array(augmented_feature_datas)
labels = np.array(augmented_labelss)
scaler = MinMaxScaler()

scaler.fit(data)
normalized_data = scaler.transform(data)
normalized_cross_data = scaler.transform(val_data)
normalized_test_data = scaler.transform(test_data)

model_xgb.fit(normalized_data, augmented_labelss)

label_predict_xgb = model_xgb.predict(normalized_cross_data)
label_predict_xgb_test = model_xgb.predict(normalized_test_data)

df = pd.DataFrame()
df1 = pd.DataFrame()

df['真实值'] = [i[0] for i in val_labels]
df1['真实值'] = [i[0] for i in test_labels]

df["xgb_预测"] = label_predict_xgb
df1["xgb_预测"] = label_predict_xgb_test

l3 = s(df['xgb_预测'], df['真实值'])
l4 = s(df1['xgb_预测'], df1['真实值'])
df["xgb_误差率"] = l3
df1["xgb_误差率"] = l4

df.to_csv("XGB-Outputs-Unsaturated concentration-test.csv")
df1.to_csv("XGB-Outputs-Unsaturated concentration-val.csv")

print(df)
print(df1)
