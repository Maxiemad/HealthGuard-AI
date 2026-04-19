import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

class MultiDiseaseEnsemble:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.metrics = {}
        self.explainers = {}
        
    def load_and_preprocess_diabetes(self):
        """Load and preprocess diabetes dataset"""
        df = pd.read_csv('data/diabetes.csv')
        
        # Handle missing values (zeros represent missing)
        zero_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_missing_cols:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
        
        # Feature engineering
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['Glucose_Category'] = pd.cut(df['Glucose'],
                                       bins=[0, 100, 126, 300],
                                       labels=['Normal', 'Prediabetic', 'Diabetic'])
        df['Age_Group'] = pd.cut(df['Age'],
                                bins=[0, 35, 50, 100],
                                labels=['Young', 'Middle', 'Senior'])
        
        # One-hot encoding
        df = pd.get_dummies(df, columns=['BMI_Category', 'Glucose_Category', 'Age_Group'])
        
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        
        return X, y
    
    def load_and_preprocess_heart(self):
        """Load and preprocess heart disease dataset"""
        df = pd.read_csv('data/heart/heart.csv')
        
        # Convert target to binary (0 = no disease, 1 = disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Handle missing values if any
        df = df.dropna()
        
        # Feature engineering
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 40, 55, 100],
                                labels=['Young', 'Middle', 'Senior'])
        df['bp_category'] = pd.cut(df['trestbps'],
                                  bins=[0, 120, 140, 200],
                                  labels=['Normal', 'Elevated', 'High'])
        df['chol_category'] = pd.cut(df['chol'],
                                     bins=[0, 200, 240, 600],
                                     labels=['Normal', 'Borderline', 'High'])
        
        # One-hot encoding
        df = pd.get_dummies(df, columns=['age_group', 'bp_category', 'chol_category'])
        
        X = df.drop("target", axis=1)
        y = df["target"]
        
        return X, y
    
    def load_and_preprocess_kidney(self):
        """Load and preprocess kidney disease dataset"""
        df = pd.read_csv('data/kidney/kidney_disease.csv')
        
        # Convert target to binary
        df['class'] = (df['class'] == 'ckd').astype(int)
        
        # Handle categorical variables
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'apppe', 'ane']
        label_encoders = {}
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'class':
                df[col] = df[col].fillna(df[col].median())
        
        X = df.drop("class", axis=1)
        y = df["class"]
        
        return X, y
    
    def train_disease_models(self, disease_name, X, y):
        """Train ensemble models for a specific disease"""
        print(f"\nTraining models for {disease_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with class balancing
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced', 
                max_iter=1000, 
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                class_weight='balanced', 
                n_estimators=100, 
                random_state=42
            ),
            'XGBoost': XGBClassifier(
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), 
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {}
        
        for name, model in models.items():
            scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=cv, scoring='roc_auc')
            cv_scores[name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
            print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Select best model
        best_model_name = max(cv_scores.keys(), key=lambda k: cv_scores[k]['mean'])
        best_model = models[best_model_name]
        
        # Train best model on full training set
        best_model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'best_model': best_model_name,
            'cv_scores': cv_scores
        }
        
        # Create SHAP explainer
        if best_model_name == 'Logistic Regression':
            explainer = shap.LinearExplainer(best_model, X_train_scaled)
        elif best_model_name == 'Random Forest':
            explainer = shap.TreeExplainer(best_model)
        else:  # XGBoost
            explainer = shap.TreeExplainer(best_model)
        
        # Store artifacts
        self.models[disease_name] = best_model
        self.scalers[disease_name] = scaler
        self.feature_names[disease_name] = list(X.columns)
        self.metrics[disease_name] = metrics
        self.explainers[disease_name] = explainer
        
        print(f"Best model for {disease_name}: {best_model_name}")
        print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}")
        
        return metrics
    
    def train_all_models(self):
        """Train models for all diseases"""
        diseases = {
            'diabetes': self.load_and_preprocess_diabetes,
            'heart': self.load_and_preprocess_heart,
            'kidney': self.load_and_preprocess_kidney
        }
        
        all_metrics = {}
        
        for disease_name, loader in diseases.items():
            try:
                X, y = loader()
                metrics = self.train_disease_models(disease_name, X, y)
                all_metrics[disease_name] = metrics
            except Exception as e:
                print(f"Error training {disease_name}: {str(e)}")
                continue
        
        return all_metrics
    
    def save_artifacts(self):
        """Save all model artifacts"""
        os.makedirs("models", exist_ok=True)
        
        # Save models
        for disease_name, model in self.models.items():
            joblib.dump(model, f"models/{disease_name}_model.pkl")
        
        # Save scalers
        for disease_name, scaler in self.scalers.items():
            joblib.dump(scaler, f"models/{disease_name}_scaler.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, "models/feature_names.pkl")
        
        # Save explainers
        for disease_name, explainer in self.explainers.items():
            joblib.dump(explainer, f"models/{disease_name}_explainer.pkl")
        
        # Save metrics
        with open("models/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        print("All artifacts saved successfully!")
    
    def predict_risk(self, disease_name, patient_data):
        """Predict risk for a specific disease"""
        if disease_name not in self.models:
            raise ValueError(f"No model trained for {disease_name}")
        
        model = self.models[disease_name]
        scaler = self.scalers[disease_name]
        
        # Ensure patient_data has correct feature order
        features = self.feature_names[disease_name]
        patient_df = pd.DataFrame([patient_data], columns=features)
        
        # Scale features
        patient_scaled = scaler.transform(patient_df)
        
        # Predict
        risk_score = model.predict_proba(patient_scaled)[0][1]
        
        return risk_score

if __name__ == "__main__":
    # Initialize ensemble trainer
    ensemble = MultiDiseaseEnsemble()
    
    # Train all models
    metrics = ensemble.train_all_models()
    
    # Save artifacts
    ensemble.save_artifacts()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for disease, disease_metrics in metrics.items():
        print(f"\n{disease.upper()}:")
        print(f"  Best Model: {disease_metrics['best_model']}")
        print(f"  Accuracy: {disease_metrics['accuracy']:.3f}")
        print(f"  ROC-AUC: {disease_metrics['roc_auc']:.3f}")
        print(f"  Recall: {disease_metrics['recall']:.3f}")
