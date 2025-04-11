import joblib  # 用于加载保存的模型
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
from rdkit.Chem import RDKFingerprint
from sklearn.preprocessing import StandardScaler  # 确保 scaler 已正确加载

# 加载数据
file_path = "your_dataset.csv"
data = pd.read_csv(file_path)

# 检查是否包含 "SMILES" 列
if "SMILES" not in data.columns:
    raise ValueError("数据集中未找到 'SMILES' 列，请确认文件格式。")

# 初始化一个列表存储Morgan描述符
morgan_descriptors = []

# 遍历SMILES列生成Morgan描述符
for smile in data["SMILES"]:
    mol = Chem.MolFromSmiles(smile)
    if mol:
        # 使用旧方法生成Morgan指纹 (半径=2, 位向量长度=2048)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        # 将位向量转换为列表
        morgan_descriptors.append(list(fp))
    else:
        morgan_descriptors.append([None] * 2048)  # 处理解析失败的SMILES

# 将描述符添加到DataFrame
morgan_df = pd.DataFrame(morgan_descriptors, columns=[f"Morgan_{i}" for i in range(2048)])
result = pd.concat([data, morgan_df], axis=1)

# 保存结果到新文件
output_file = "Morgan.csv"
result.to_csv(output_file, index=False)


# 读取 PaDEL 生成的描述符文件
moragn_descriptors = pd.read_csv("Morgan.csv")

# 确保与训练数据的列保持一致
X_moragn = moragn_descriptors.drop(columns=['Name', 'SMILES'], errors='ignore')  # 移除非描述符列

# 确保 `scaler` 是在模型训练时保存的
scaler = joblib.load("scaler_morgan_svm.pkl")  # 加载训练时保存的 scaler


# 加载已保存的模型
best_svc = joblib.load("svm_morgan_model.pkl")

# 加载训练时的特征名
train_features = scaler.feature_names_in_  # 从 scaler 提取训练时的特征名

# 筛选新数据，只保留训练时存在的特征
X_moragn_filtered = X_moragn.loc[:, X_moragn.columns.isin(train_features)]

# 检查筛选后的特征是否完全与训练时一致
X_moragn_aligned = X_moragn_filtered.reindex(columns=train_features, fill_value=0)

# 标准化筛选后的数据
X_moragn_scaled = scaler.transform(X_moragn_aligned)

# 预测活性
svm_predictions = best_svc.predict(X_moragn_scaled)
svm_probabilities = best_svc.predict_proba(X_moragn_scaled)[:, 1]

# 将预测结果与原始数据结合
moragn_descriptors['Predicted Activity'] = svm_predictions
moragn_descriptors['Probability'] = svm_probabilities

# 保存预测结果到 Excel
moragn_descriptors.to_csv("svm_predictions.csv", index=False)


# Step 2: 确保DataFrame中包含SMILES列
if 'SMILES' not in data.columns:
    raise ValueError("DataFrame中找不到SMILES列！请确保列名正确。")

# Step 3: 生成FP2指纹并将其转化为多列
def get_fp2_bits(smiles):
    """根据SMILES字符串生成FP2指纹并转化为一个二进制向量"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp2 = RDKFingerprint(mol)
    return list(fp2)

# 生成FP2指纹并转换为列
fp2_bits = data['SMILES'].apply(get_fp2_bits)

# 获取最大位数，填充不足的部分（确保所有行有相同长度的指纹）
max_len = max(fp2_bits.apply(len))
fp2_bits_padded = fp2_bits.apply(lambda x: x + [0] * (max_len - len(x)) if x is not None else [0] * max_len)

# 将FP2指纹转为多列
fp2_df = pd.DataFrame(fp2_bits_padded.tolist(), columns=[f'FP2_{i}' for i in range(max_len)])

# Step 4: 合并原始数据和FP2指纹
df_with_fp2 = pd.concat([data, fp2_df], axis=1)

# Step 5: 将结果保存到新的Excel文件
output_file = 'RDKfp.csv'  # 输出文件名
df_with_fp2.to_csv(output_file, index=False)



# 读取 PaDEL 生成的描述符文件
RDKfp_descriptors = pd.read_csv("RDKfp.csv")

# 确保与训练数据的列保持一致
X_RDKfp = RDKfp_descriptors.drop(columns=['Name', 'SMILES'], errors='ignore')  # 移除非描述符列

# 确保 `scaler` 是在模型训练时保存的
scaler = joblib.load("scaler_RDKfp_rf.pkl")  # 加载训练时保存的 scaler


# 加载已保存的模型
best_rf_model = joblib.load("rf_RDKfp_model.pkl")


# 加载训练时的特征名
train_features = scaler.feature_names_in_  # 从 scaler 提取训练时的特征名

# 筛选新数据，只保留训练时存在的特征
X_RDKfp_filtered = X_RDKfp.loc[:, X_RDKfp.columns.isin(train_features)]

# 检查筛选后的特征是否完全与训练时一致
X_RDKfp_aligned = X_RDKfp_filtered.reindex(columns=train_features, fill_value=0)

# 标准化筛选后的数据
X_RDKfp_scaled = scaler.transform(X_RDKfp_aligned)

# 预测活性
RF_predictions = best_rf_model.predict(X_RDKfp_scaled)
RF_probabilities = best_rf_model.predict_proba(X_RDKfp_scaled)[:, 1]

# 将预测结果与原始数据结合
RDKfp_descriptors['Predicted Activity'] = RF_predictions
RDKfp_descriptors['Probability'] = RF_probabilities

# 保存预测结果到 Excel
RDKfp_descriptors.to_csv("RF_predictions.csv", index=False)


# 假设SMILES列的名称是 'SMILES'，EC50列的名称是 'EC50'
smiles = data['SMILES']

# 2. 将SMILES列保存到.smi文件供PaDEL使用
with open('molecules.smi', 'w') as f:
    for smile in smiles:
        f.write(smile + '\n')

# PaDEL JAR 文件路径
padel_jar_path = '/your/path/to/PaDEL-Descriptor.jar'

# 输入和输出文件路径
input_file = 'molecules.smi'
output_file = 'PaDEL_descriptors.csv'  # PaDEL 通常生成 .csv 文件

# 3. 使用 PaDEL 生成 2D 描述符
command = f'java -jar "{padel_jar_path}" -dir "{input_file}" -file "{output_file}" -2d'

# 执行命令
subprocess.run(command, shell=True, check=True)



# 4. 读取PaDEL生成的描述符文件
paDEL_descriptors = pd.read_csv("PaDEL_descriptors.csv")


# 5. 按数字排序Name列
# 提取出AUTOGEN_molecules_x中的数字部分并进行排序
paDEL_descriptors['Name_number'] = paDEL_descriptors['Name'].str.extract('(\d+)').astype(int)

# 按数字顺序排序数据
paDEL_descriptors = paDEL_descriptors.sort_values(by='Name_number').reset_index(drop=True)

# 6. 将PaDEL生成的描述符与原始数据（SMILES）进行合并
# 保证SMILES列与描述符列一一对应
paDEL_descriptors['SMILES'] = data['SMILES']  # 保留原始 SMILES 列

# 7. 删除临时的Name_number列
paDEL_descriptors = paDEL_descriptors.drop(columns=['Name_number'])

# 8. 保存合并后的结果到新的CSV文件
output_file_with_descriptors = 'PaDEL.csv'  # 输出文件名
paDEL_descriptors.to_csv(output_file_with_descriptors, index=False)



# 读取 PaDEL 生成的描述符文件
PaDEL_descriptors = pd.read_csv("PaDEL.csv")


# 确保与训练数据的列保持一致
X_PaDEL = PaDEL_descriptors.drop(columns=['Name', 'SMILES'], errors='ignore')  # 移除非描述符列

# 确保 `scaler` 是在模型训练时保存的
scaler = joblib.load("scaler_padel_xgboost.pkl")  # 加载训练时保存的 scaler


# 加载已保存的模型
best_xgb = joblib.load("xgboost_padel_model.pkl")


# 加载训练时的特征名
train_features = scaler.feature_names_in_  # 从 scaler 提取训练时的特征名

# 筛选新数据，只保留训练时存在的特征
X_PaDEL_filtered = X_PaDEL.loc[:, X_PaDEL.columns.isin(train_features)]

# 检查筛选后的特征是否完全与训练时一致
X_PaDEL_aligned = X_PaDEL_filtered.reindex(columns=train_features, fill_value=0)

# 标准化筛选后的数据
X_PaDEL_scaled = scaler.transform(X_PaDEL_aligned)

# 预测活性
xgboost_predictions = best_xgb.predict(X_PaDEL_scaled)
xgboost_probabilities = best_xgb.predict_proba(X_PaDEL_scaled)[:, 1]

# 将预测结果与原始数据结合
PaDEL_descriptors['Predicted Activity'] = xgboost_predictions
PaDEL_descriptors['Probability'] = xgboost_probabilities

# 保存预测结果到 Excel
PaDEL_descriptors.to_csv("xgboost_predictions.csv", index=False)



# 定义模型文件与对应的列名
models = [
    ("svm_predictions.csv", "SVM_Prediction"),
    ("RF_predictions.csv", "RF_Prediction"),
    ("xgboost_predictions.csv", "XGB_Prediction")
]

# 使用reduce进行链式合并
from functools import reduce
merged = reduce(
    lambda left, right: left.merge(right, on="SMILES"),
    [pd.read_csv(f).rename(columns={"Predicted Activity": name})[["SMILES", name]]
     for f, name in models]
)

# 筛选并保存结果
merged.query("SVM_Prediction + RF_Prediction + XGB_Prediction >= 2") \
      .to_csv("filtered_results.csv", index=False)

