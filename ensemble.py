import joblib 
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
from rdkit.Chem import RDKFingerprint
from sklearn.preprocessing import StandardScaler  


file_path = "your_dataset.csv"
data = pd.read_csv(file_path)


if "SMILES" not in data.columns:
    raise ValueError("数据集中未找到 'SMILES' 列，请确认文件格式。")

morgan_descriptors = []

for smile in data["SMILES"]:
    mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        morgan_descriptors.append(list(fp))
    else:
        morgan_descriptors.append([None] * 2048) 


morgan_df = pd.DataFrame(morgan_descriptors, columns=[f"Morgan_{i}" for i in range(2048)])
result = pd.concat([data, morgan_df], axis=1)

output_file = "Morgan.csv"
result.to_csv(output_file, index=False)



moragn_descriptors = pd.read_csv("Morgan.csv")


X_moragn = moragn_descriptors.drop(columns=['Name', 'SMILES'], errors='ignore')  # 移除非描述符列


scaler = joblib.load("scaler_morgan_svm.pkl")  


best_svc = joblib.load("svm_morgan_model.pkl")

train_features = scaler.feature_names_in_  


X_moragn_filtered = X_moragn.loc[:, X_moragn.columns.isin(train_features)]


X_moragn_aligned = X_moragn_filtered.reindex(columns=train_features, fill_value=0)


X_moragn_scaled = scaler.transform(X_moragn_aligned)


svm_predictions = best_svc.predict(X_moragn_scaled)
svm_probabilities = best_svc.predict_proba(X_moragn_scaled)[:, 1]


moragn_descriptors['Predicted Activity'] = svm_predictions
moragn_descriptors['Probability'] = svm_probabilities


moragn_descriptors.to_csv("svm_predictions.csv", index=False)



if 'SMILES' not in data.columns:
    raise ValueError("DataFrame中找不到SMILES列！请确保列名正确。")


def get_fp2_bits(smiles):
    """根据SMILES字符串生成FP2指纹并转化为一个二进制向量"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp2 = RDKFingerprint(mol)
    return list(fp2)


fp2_bits = data['SMILES'].apply(get_fp2_bits)


max_len = max(fp2_bits.apply(len))
fp2_bits_padded = fp2_bits.apply(lambda x: x + [0] * (max_len - len(x)) if x is not None else [0] * max_len)

fp2_df = pd.DataFrame(fp2_bits_padded.tolist(), columns=[f'FP2_{i}' for i in range(max_len)])


df_with_fp2 = pd.concat([data, fp2_df], axis=1)


output_file = 'RDKfp.csv' 
df_with_fp2.to_csv(output_file, index=False)


RDKfp_descriptors = pd.read_csv("RDKfp.csv")


X_RDKfp = RDKfp_descriptors.drop(columns=['Name', 'SMILES'], errors='ignore')  # 移除非描述符列


scaler = joblib.load("scaler_RDKfp_rf.pkl") 


best_rf_model = joblib.load("rf_RDKfp_model.pkl")


train_features = scaler.feature_names_in_  # 从 scaler 提取训练时的特征名


X_RDKfp_filtered = X_RDKfp.loc[:, X_RDKfp.columns.isin(train_features)]

X_RDKfp_aligned = X_RDKfp_filtered.reindex(columns=train_features, fill_value=0)

X_RDKfp_scaled = scaler.transform(X_RDKfp_aligned)

RF_predictions = best_rf_model.predict(X_RDKfp_scaled)
RF_probabilities = best_rf_model.predict_proba(X_RDKfp_scaled)[:, 1]

RDKfp_descriptors['Predicted Activity'] = RF_predictions
RDKfp_descriptors['Probability'] = RF_probabilities

RDKfp_descriptors.to_csv("RF_predictions.csv", index=False)

smiles = data['SMILES']


with open('molecules.smi', 'w') as f:
    for smile in smiles:
        f.write(smile + '\n')


padel_jar_path = '/your/path/to/PaDEL-Descriptor.jar'


input_file = 'molecules.smi'
output_file = 'PaDEL_descriptors.csv'  


command = f'java -jar "{padel_jar_path}" -dir "{input_file}" -file "{output_file}" -2d'

subprocess.run(command, shell=True, check=True)

paDEL_descriptors = pd.read_csv("PaDEL_descriptors.csv")


paDEL_descriptors['Name_number'] = paDEL_descriptors['Name'].str.extract('(\d+)').astype(int)


paDEL_descriptors = paDEL_descriptors.sort_values(by='Name_number').reset_index(drop=True)

paDEL_descriptors['SMILES'] = data['SMILES'] 

paDEL_descriptors = paDEL_descriptors.drop(columns=['Name_number'])

output_file_with_descriptors = 'PaDEL.csv'  
paDEL_descriptors.to_csv(output_file_with_descriptors, index=False)

PaDEL_descriptors = pd.read_csv("PaDEL.csv")

X_PaDEL = PaDEL_descriptors.drop(columns=['Name', 'SMILES'], errors='ignore') 
scaler = joblib.load("scaler_padel_xgboost.pkl")  

best_xgb = joblib.load("xgboost_padel_model.pkl")

train_features = scaler.feature_names_in_  

X_PaDEL_filtered = X_PaDEL.loc[:, X_PaDEL.columns.isin(train_features)]

X_PaDEL_aligned = X_PaDEL_filtered.reindex(columns=train_features, fill_value=0)

X_PaDEL_scaled = scaler.transform(X_PaDEL_aligned)

xgboost_predictions = best_xgb.predict(X_PaDEL_scaled)
xgboost_probabilities = best_xgb.predict_proba(X_PaDEL_scaled)[:, 1]

PaDEL_descriptors['Predicted Activity'] = xgboost_predictions
PaDEL_descriptors['Probability'] = xgboost_probabilities

PaDEL_descriptors.to_csv("xgboost_predictions.csv", index=False)


models = [
    ("svm_predictions.csv", "SVM_Prediction"),
    ("RF_predictions.csv", "RF_Prediction"),
    ("xgboost_predictions.csv", "XGB_Prediction")
]

from functools import reduce
merged = reduce(
    lambda left, right: left.merge(right, on="SMILES"),
    [pd.read_csv(f).rename(columns={"Predicted Activity": name})[["SMILES", name]]
     for f, name in models]
)

merged.query("SVM_Prediction + RF_Prediction + XGB_Prediction >= 2") \
      .to_csv("filtered_results.csv", index=False)

