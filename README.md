# Ensemble--β2AR
The machine learning-based screening model for β2AR agonists.

This project requires the following Python libraries and their specific versions:
- joblib: 1.4.2
- pandas: 2.2.3
- rdkit: 2024.3.6
- subprocess32: 3.5.4

Morgan_svm_model.pkl, PaDEL_xgboost_model.pkl, RDKfp_RF_model.pkl are trained models.
Morgan_svm_scaler.pkl, PaDEL_xgboost_scaler.pkl, RDKfp_RF_scaler.pkl are trained normalizers that convert the input fingerprint data into a suitable normalized form to ensure that the model can accurately predict the new data.
