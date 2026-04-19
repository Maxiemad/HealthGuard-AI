import numpy as np
import pandas as pd
import joblib
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TemporalRiskProjector:
    def __init__(self):
        # Load trained models
        self.models = {}
        self.scalers = {}
        self.feature_names = joblib.load("models/feature_names.pkl")
        
        diseases = ['diabetes', 'heart', 'kidney']
        for disease in diseases:
            try:
                self.models[disease] = joblib.load(f"models/{disease}_model.pkl")
                self.scalers[disease] = joblib.load(f"models/{disease}_scaler.pkl")
            except:
                print(f"Warning: Could not load {disease} model")
    
    def project_forward(self, disease_name, current_features, months_ahead, intervention=None):
        """
        Projects risk N months ahead with optional intervention
        
        Args:
            disease_name: 'diabetes', 'heart', or 'kidney'
            current_features: dict of current patient features
            months_ahead: number of months to project
            intervention: dict with intervention parameters
        """
        if disease_name not in self.models:
            raise ValueError(f"No model available for {disease_name}")
        
        model = self.models[disease_name]
        scaler = self.scalers[disease_name]
        
        # Start with current features
        projected_features = current_features.copy()
        
        # Natural progression over time
        years_ahead = months_ahead / 12
        
        if disease_name == 'diabetes':
            # Age increases
            if 'Age' in projected_features:
                projected_features['Age'] += years_ahead
            
            # Natural metabolic degradation
            if 'Glucose' in projected_features:
                projected_features['Glucose'] += years_ahead * 2  # +2 mg/dL per year
            
            if 'BloodPressure' in projected_features:
                projected_features['BloodPressure'] += years_ahead * 1  # +1 mmHg per year
            
            if 'BMI' in projected_features:
                projected_features['BMI'] += years_ahead * 0.1  # Slight BMI increase
            
        elif disease_name == 'heart':
            # Age progression
            if 'age' in projected_features:
                projected_features['age'] += years_ahead
            
            # Blood pressure tends to increase with age
            if 'trestbps' in projected_features:
                projected_features['trestbps'] += years_ahead * 1.5
            
            # Cholesterol may increase slightly
            if 'chol' in projected_features:
                projected_features['chol'] += years_ahead * 2
        
        elif disease_name == 'kidney':
            # Age progression
            if 'age' in projected_features:
                projected_features['age'] += years_ahead
            
            # Blood pressure impact on kidney
            if 'bp' in projected_features:
                projected_features['bp'] += years_ahead * 1.2
            
            # GFR tends to decrease with age (represented by higher creatinine)
            if 'sc' in projected_features:  # serum creatinine
                projected_features['sc'] += years_ahead * 0.02
        
        # Apply intervention effects
        if intervention:
            if disease_name == 'diabetes':
                if 'bmi_reduction' in intervention:
                    bmi_change = intervention['bmi_reduction']
                    projected_features['BMI'] -= bmi_change
                    
                    # BMI reduction improves glucose and BP
                    if 'Glucose' in projected_features:
                        projected_features['Glucose'] -= bmi_change * 1.5
                    if 'BloodPressure' in projected_features:
                        projected_features['BloodPressure'] -= bmi_change * 0.8
                
                if 'exercise_program' in intervention and intervention['exercise_program']:
                    # Exercise improves all metabolic markers
                    if 'Glucose' in projected_features:
                        projected_features['Glucose'] -= 5
                    if 'BloodPressure' in projected_features:
                        projected_features['BloodPressure'] -= 3
                    if 'BMI' in projected_features:
                        projected_features['BMI'] -= 0.5
            
            elif disease_name == 'heart':
                if 'exercise_program' in intervention and intervention['exercise_program']:
                    if 'trestbps' in projected_features:
                        projected_features['trestbps'] -= 5
                    if 'chol' in projected_features:
                        projected_features['chol'] -= 10
                
                if 'diet_improvement' in intervention and intervention['diet_improvement']:
                    if 'chol' in projected_features:
                        projected_features['chol'] -= 15
                    if 'trestbps' in projected_features:
                        projected_features['trestbps'] -= 3
        
        # Bootstrap for uncertainty quantification
        risk_samples = []
        
        # Prepare features for model
        features_df = self._prepare_features(projected_features, disease_name)
        
        if features_df is None:
            return {'mean_risk': 0.5, 'ci_lower': 0.4, 'ci_upper': 0.6}
        
        for _ in range(100):  # 100 bootstrap samples
            # Add noise to simulate uncertainty
            noisy_features = features_df.copy()
            for col in noisy_features.columns:
                if noisy_features[col].dtype in ['float64', 'int64']:
                    noise_std = abs(noisy_features[col].iloc[0]) * 0.05  # 5% noise
                    noisy_features[col] += np.random.normal(0, noise_std)
            
            # Scale features
            try:
                scaled_features = scaler.transform(noisy_features)
                risk = model.predict_proba(scaled_features)[0][1]
                risk_samples.append(risk)
            except:
                risk_samples.append(0.5)  # Default if prediction fails
        
        if not risk_samples:
            return {'mean_risk': 0.5, 'ci_lower': 0.4, 'ci_upper': 0.6}
        
        return {
            'mean_risk': np.mean(risk_samples),
            'ci_lower': np.percentile(risk_samples, 2.5),
            'ci_upper': np.percentile(risk_samples, 97.5),
            'projected_features': projected_features
        }
    
    def _prepare_features(self, features_dict, disease_name):
        """Prepare features for the specific disease model"""
        try:
            # Get expected feature names for this disease
            expected_features = self.feature_names[disease_name]
            
            # Create DataFrame with all required features
            patient_data = {}
            for feature in expected_features:
                if feature in features_dict:
                    patient_data[feature] = features_dict[feature]
                else:
                    # Set default value for missing features
                    patient_data[feature] = 0
            
            return pd.DataFrame([patient_data])
        except:
            return None
    
    def create_timeline_chart(self, disease_name, current_features, interventions=None):
        """Create an interactive timeline chart showing risk projection"""
        
        # Time points (current, 6 months, 1 year, 2 years)
        time_points = [0, 6, 12, 24]
        
        # Baseline projection (no intervention)
        baseline_risks = []
        for months in time_points:
            result = self.project_forward(disease_name, current_features, months)
            baseline_risks.append(result['mean_risk'])
        
        # Intervention projections
        intervention_risks = {}
        if interventions:
            for intervention_name, intervention_params in interventions.items():
                risks = []
                for months in time_points:
                    result = self.project_forward(
                        disease_name, current_features, months, intervention_params
                    )
                    risks.append(result['mean_risk'])
                intervention_risks[intervention_name] = risks
        
        # Create plot
        fig = go.Figure()
        
        # Add baseline
        fig.add_trace(go.Scatter(
            x=time_points,
            y=[r * 100 for r in baseline_risks],
            mode='lines+markers',
            name='No Intervention',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # Add confidence interval for baseline
        ci_lower = []
        ci_upper = []
        for months in time_points:
            result = self.project_forward(disease_name, current_features, months)
            ci_lower.append(result['ci_lower'] * 100)
            ci_upper.append(result['ci_upper'] * 100)
        
        fig.add_trace(go.Scatter(
            x=time_points + time_points[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Confidence Interval'
        ))
        
        # Add intervention lines
        colors = ['green', 'blue', 'purple']
        for i, (intervention_name, risks) in enumerate(intervention_risks.items()):
            fig.add_trace(go.Scatter(
                x=time_points,
                y=[r * 100 for r in risks],
                mode='lines+markers',
                name=intervention_name,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                marker=dict(size=6)
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{disease_name.title()} Risk Projection Over Time',
            xaxis_title='Time (months)',
            yaxis_title='Risk Score (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        # Add risk zones
        fig.add_hrect(y0=0, y1=40, fillcolor="lightgreen", opacity=0.3, layer="below", line_width=0)
        fig.add_hrect(y0=40, y1=70, fillcolor="yellow", opacity=0.3, layer="below", line_width=0)
        fig.add_hrect(y0=70, y1=100, fillcolor="lightcoral", opacity=0.3, layer="below", line_width=0)
        
        return fig
    
    def get_risk_category(self, risk_score):
        """Categorize risk score"""
        if risk_score < 0.4:
            return "Low Risk", "green"
        elif risk_score < 0.7:
            return "Moderate Risk", "orange"
        else:
            return "High Risk", "red"

class InterventionSimulator:
    def __init__(self):
        self.projector = TemporalRiskProjector()
    
    def simulate_interventions(self, disease_name, current_features):
        """Simulate various interventions and their effects"""
        
        interventions = {
            'Lifestyle Change': {
                'bmi_reduction': 2.0,
                'exercise_program': True
            },
            'Aggressive Lifestyle': {
                'bmi_reduction': 4.0,
                'exercise_program': True
            },
            'Exercise Only': {
                'exercise_program': True
            }
        }
        
        # Get current risk
        current_result = self.projector.project_forward(disease_name, current_features, 0)
        current_risk = current_result['mean_risk']
        
        # Calculate 6-month projections for each intervention
        results = {}
        for intervention_name, params in interventions.items():
            future_result = self.projector.project_forward(
                disease_name, current_features, 6, params
            )
            results[intervention_name] = {
                'current_risk': current_risk,
                'future_risk': future_result['mean_risk'],
                'risk_reduction': current_risk - future_result['mean_risk'],
                'ci_lower': future_result['ci_lower'],
                'ci_upper': future_result['ci_upper']
            }
        
        return results

if __name__ == "__main__":
    # Test the temporal projector
    projector = TemporalRiskProjector()
    
    # Example patient data (diabetes)
    patient_data = {
        'Pregnancies': 2,
        'Glucose': 145,
        'BloodPressure': 80,
        'SkinThickness': 20,
        'Insulin': 85,
        'BMI': 32.5,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 45,
        'BMI_Category_Obese': 1,
        'BMI_Category_Overweight': 0,
        'BMI_Category_Normal': 0,
        'BMI_Category_Underweight': 0,
        'Glucose_Category_Diabetic': 1,
        'Glucose_Category_Prediabetic': 0,
        'Glucose_Category_Normal': 0,
        'Age_Group_Senior': 1,
        'Age_Group_Middle': 0,
        'Age_Group_Young': 0
    }
    
    # Test projection
    result = projector.project_forward('diabetes', patient_data, 6)
    print(f"6-month projection: {result['mean_risk']:.3f} ± {result['ci_upper'] - result['ci_lower']:.3f}")
    
    # Test intervention simulation
    simulator = InterventionSimulator()
    interventions = simulator.simulate_interventions('diabetes', patient_data)
    
    for name, data in interventions.items():
        print(f"{name}: {data['risk_reduction']:.3f} risk reduction")
