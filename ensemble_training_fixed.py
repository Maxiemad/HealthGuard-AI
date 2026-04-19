import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import shap
import json
import os

class FixedMultiDiseaseEnsemble:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.metrics = {}
        self.explainers = {}
    
    def load_and_preprocess_diabetes(self):
        """Load and preprocess diabetes dataset with improved feature engineering"""
        df = pd.read_csv('data/diabetes.csv')
        
        # Replace 0 values with NaN for proper handling
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            df[col] = df[col].replace(0, np.nan)
        
        # Handle missing values with median
        for col in zero_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Enhanced feature engineering
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 25, 30, 50],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['Glucose_Category'] = pd.cut(df['Glucose'], 
                                      bins=[0, 100, 126, 200],
                                      labels=['Normal', 'Prediabetic', 'Diabetic'])
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[0, 35, 50, 100],
                                labels=['Young', 'Middle', 'Senior'])
        
        # Add interaction features
        df['BMI_Glucose_Interaction'] = df['BMI'] * df['Glucose']
        df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
        df['Glucose_Age_Interaction'] = df['Glucose'] * df['Age']
        
        # One-hot encoding
        df = pd.get_dummies(df, columns=['BMI_Category', 'Glucose_Category', 'Age_Group'])
        
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        return X, y
    
    def load_and_preprocess_heart(self):
        """Load and preprocess heart disease dataset with improved features"""
        df = pd.read_csv('data/heart/heart.csv')
        
        # Binarize target (0 = no disease, >0 = disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Enhanced feature engineering
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 40, 55, 100],
                                labels=['Young', 'Middle', 'Senior'])
        df['bp_category'] = pd.cut(df['trestbps'],
                                  bins=[0, 120, 140, 200],
                                  labels=['Normal', 'Elevated', 'High'])
        df['chol_category'] = pd.cut(df['chol'],
                                     bins=[0, 200, 240, 600],
                                     labels=['Normal', 'Borderline', 'High'])
        
        # Add interaction features
        df['age_chol_interaction'] = df['age'] * df['chol']
        df['thalach_slope_interaction'] = df['thalach'] * df['slope']
        df['age_bp_interaction'] = df['age'] * df['trestbps']
        
        # One-hot encoding
        df = pd.get_dummies(df, columns=['age_group', 'bp_category', 'chol_category'])
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        return X, y
    
    def load_and_preprocess_kidney(self):
        """Load and preprocess kidney disease dataset with only numeric features"""
        df = pd.read_csv('data/kidney/kidney_disease_clean.csv')
        
        # Convert target to binary
        df['class'] = (df['class'] == 'ckd').astype(int)
        
        print(f"Kidney dataset shape: {df.shape}")
        print(f"Kidney target distribution: {df['class'].value_counts().to_dict()}")
        
        # Handle missing values - only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'class':
                df[col] = df[col].fillna(df[col].median())
        
        X = df.drop("class", axis=1)
        y = df["class"]
        
        return X, y
    
    def train_disease_model(self, disease_name):
        """Train improved models for a specific disease with proper handling"""
        
        # Load and preprocess data
        if disease_name == 'diabetes':
            X, y = self.load_and_preprocess_diabetes()
        elif disease_name == 'heart':
            X, y = self.load_and_preprocess_heart()
        elif disease_name == 'kidney':
            X, y = self.load_and_preprocess_kidney()
        else:
            raise ValueError(f"Unknown disease: {disease_name}")
        
        print(f"\n=== Training {disease_name.upper()} models ===")
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert y_train to numpy array for consistent indexing
        y_train = np.array(y_train)
        
        # Apply SMOTE only for larger datasets (skip for small heart/kidney datasets)
        if disease_name == 'diabetes':
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE - Training shape: {X_train_resampled.shape}")
            print(f"After SMOTE - Target distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
            X_train_scaled = X_train_resampled
            y_train = y_train_resampled
        else:
            print(f"Using original training data (no SMOTE for {disease_name})")
        
        # Define models with class_weight='balanced'
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000, 
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42, 
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42, 
                eval_metric='logloss',
                scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
            )
        }
        
        # Hyperparameter grids for GridSearchCV
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        cv_scores = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name in param_grids:
                # Grid search for tree models
                grid_search = GridSearchCV(
                    model, 
                    param_grids[name], 
                    cv=cv, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                print(f"Best params for {name}: {grid_search.best_params_}")
            else:
                # Logistic Regression with default parameters
                best_model = model
                best_model.fit(X_train_scaled, y_train)
            
            # Cross-validation score
            scores = []
            for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                best_model.fit(X_tr, y_tr)
                y_prob = best_model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, y_prob))
            
            cv_scores[name] = {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            print(f"{name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        # Select best model
        best_model_name = max(cv_scores.keys(), key=lambda k: cv_scores[k]['mean'])
        best_model = models[best_model_name]
        
        # Retrain best model on full training set with best params if applicable
        if best_model_name in param_grids:
            grid_search = GridSearchCV(
                best_model, 
                param_grids[best_model_name], 
                cv=cv, 
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
        else:
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
        
        print(f"\nBest model for {disease_name}: {best_model_name}")
        print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"Test Accuracy: {metrics['accuracy']:.3f}")
        
        return metrics
    
    def train_all_models(self):
        """Train improved models for all diseases"""
        diseases = ['diabetes', 'heart', 'kidney']
        
        for disease in diseases:
            try:
                self.train_disease_model(disease)
            except Exception as e:
                print(f"Error training {disease}: {e}")
        
        # Save all artifacts
        self.save_artifacts()
        
        # Print comparison table
        self.print_model_comparison()
    
    def save_artifacts(self):
        """Save all trained models and artifacts"""
        os.makedirs('models', exist_ok=True)
        
        # Save models
        for disease, model in self.models.items():
            joblib.dump(model, f"models/{disease}_model.pkl")
        
        # Save scalers
        for disease, scaler in self.scalers.items():
            joblib.dump(scaler, f"models/{disease}_scaler.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, "models/feature_names.pkl")
        
        # Save metrics
        with open("models/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save explainers
        for disease, explainer in self.explainers.items():
            joblib.dump(explainer, f"models/{disease}_explainer.pkl")
        
        print("\nAll artifacts saved to models/ directory")
    
    def print_model_comparison(self):
        """Print detailed model comparison table"""
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        
        for disease, metrics in self.metrics.items():
            print(f"\n{disease.upper()}:")
            print(f"  Best Model: {metrics['best_model']}")
            print(f"  Test ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"  Test Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Test Precision: {metrics['precision']:.3f}")
            print(f"  Test Recall: {metrics['recall']:.3f}")
            print(f"  Test F1-Score: {metrics['f1']:.3f}")
            
            print(f"\n  Cross-Validation Scores:")
            for model_name, cv_score in metrics['cv_scores'].items():
                print(f"    {model_name}: {cv_score['mean']:.3f} ± {cv_score['std']:.3f}")

if __name__ == "__main__":
    import os
    
    # Train improved models
    ensemble = FixedMultiDiseaseEnsemble()
    ensemble.train_all_models()
