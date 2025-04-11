# Ensemble--β2AR
The machine learning-based screening model for β2AR agonists.

This project requires the following Python libraries and their specific versions:
- joblib: 1.4.2
- pandas: 2.2.3
- rdkit: 2024.3.6
- subprocess32: 3.5.4

Morgan_svm_model.pkl，PaDEL_xgboost_model.pkl，RDKfp_RF_model.pkl是训练好的模型。
Morgan_svm_scaler.pkl，PaDEL_xgboost_scaler.pkl，RDKfp_RF_scaler.pkl是训练好的标准化器，用来将输入的指纹数据转换为合适的标准化形式，以确保模型可以准确地对新数据进行预测。

