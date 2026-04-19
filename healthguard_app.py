import pandas as pd
import streamlit as st
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import io
import base64

# Import our custom modules
from ensemble_training import MultiDiseaseEnsemble
from temporal_projector import TemporalRiskProjector, InterventionSimulator
from explainability import ExplainabilityTrinity
from agent.graph import get_agent
from rag.retriever import get_retriever

# Page configuration
st.set_page_config(
    page_title="HealthGuard AI - Multi-Disease Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        color: #333333;
    }
    .risk-high { border-left-color: #ff4444; background-color: #fff5f5; color: #cc0000; }
    .risk-moderate { border-left-color: #ff8800; background-color: #fff9f0; color: #cc6600; }
    .risk-low { border-left-color: #00c851; background-color: #f0fff4; color: #006600; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and components"""
    try:
        agent = get_agent()
        projector = TemporalRiskProjector()
        explainer = ExplainabilityTrinity()
        retriever = get_retriever()
        
        # Load metrics
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
        
        return agent, projector, explainer, retriever, metrics
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, {}

def create_risk_gauge(risk_score, disease_name):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{disease_name.title()} Risk"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_timeline_chart(projections, interventions):
    """Create timeline projection chart"""
    fig = go.Figure()
    
    # Time points
    time_points = [0, 6, 12, 24]
    
    # Add baseline projection
    baseline_risks = []
    for months in time_points:
        if months == 0:
            baseline_risks.append(projections.get('current', 0) * 100)
        elif months == 6:
            baseline_risks.append(projections.get('6_months', {}).get('mean_risk', 0) * 100)
        elif months == 12:
            baseline_risks.append(projections.get('1_year', {}).get('mean_risk', 0) * 100)
        else:
            baseline_risks.append(projections.get('1_year', {}).get('mean_risk', 0) * 100)  # Assume same as 1 year
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=baseline_risks,
        mode='lines+markers',
        name='No Intervention',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Add intervention lines
    colors = ['green', 'blue', 'purple']
    for i, (intervention_name, effect) in enumerate(interventions.items()):
        if 'risk_reduction' in effect:
            current_risk = projections.get('current', 0)
            future_risk = current_risk - effect['risk_reduction']
            
            # Simple projection for intervention
            intervention_risks = [
                current_risk * 100,
                future_risk * 100,
                future_risk * 100,
                future_risk * 100
            ]
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=intervention_risks,
                mode='lines+markers',
                name=intervention_name,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                marker=dict(size=6)
            ))
    
    # Update layout
    fig.update_layout(
        title='Risk Projection Over Time',
        xaxis_title='Time (months)',
        yaxis_title='Risk Score (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=40, fillcolor="lightgreen", opacity=0.3, layer="below", line_width=0)
    fig.add_hrect(y0=40, y1=70, fillcolor="yellow", opacity=0.3, layer="below", line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="lightcoral", opacity=0.3, layer="below", line_width=0)
    
    return fig

def main():
    # Load models
    agent, projector, explainer, retriever, metrics = load_models()
    
    if not agent:
        st.error("Failed to load models. Please run ensemble_training.py first.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">HealthGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Multi-Disease Risk Intelligence with Temporal Projections & Personalized Care Pathways</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## Patient Input")
    
    # Input section
    input_tab, batch_tab, about_tab = st.tabs(["Individual Assessment", "Batch Processing", "About"])
    
    with input_tab:
        # Patient demographics
        st.markdown("### Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        
        with col2:
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
        
        # Clinical measurements
        st.markdown("### Clinical Measurements")
        col3, col4 = st.columns(2)
        
        with col3:
            glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=100)
            bp = st.number_input("Blood Pressure (mmHg)", min_value=60, max_value=200, value=120)
        
        with col4:
            skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            insulin = st.number_input("Insulin (µU/mL)", min_value=0, max_value=1000, value=85)
        
        # Create patient data dictionary
        patient_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        # Add engineered features for diabetes model
        if bmi < 18.5:
            bmi_cat = 'Underweight'
        elif bmi < 25:
            bmi_cat = 'Normal'
        elif bmi < 30:
            bmi_cat = 'Overweight'
        else:
            bmi_cat = 'Obese'
        
        if glucose < 100:
            glucose_cat = 'Normal'
        elif glucose < 126:
            glucose_cat = 'Prediabetic'
        else:
            glucose_cat = 'Diabetic'
        
        if age < 35:
            age_group = 'Young'
        elif age < 50:
            age_group = 'Middle'
        else:
            age_group = 'Senior'
        
        # Add one-hot encoded features
        patient_data.update({
            'BMI_Category_Underweight': 1 if bmi_cat == 'Underweight' else 0,
            'BMI_Category_Normal': 1 if bmi_cat == 'Normal' else 0,
            'BMI_Category_Overweight': 1 if bmi_cat == 'Overweight' else 0,
            'BMI_Category_Obese': 1 if bmi_cat == 'Obese' else 0,
            'Glucose_Category_Normal': 1 if glucose_cat == 'Normal' else 0,
            'Glucose_Category_Prediabetic': 1 if glucose_cat == 'Prediabetic' else 0,
            'Glucose_Category_Diabetic': 1 if glucose_cat == 'Diabetic' else 0,
            'Age_Group_Young': 1 if age_group == 'Young' else 0,
            'Age_Group_Middle': 1 if age_group == 'Middle' else 0,
            'Age_Group_Senior': 1 if age_group == 'Senior' else 0
        })
        
        # Analyze button
        if st.button("Comprehensive Risk Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                # Get risk summary
                risk_summary = agent.get_risk_summary(patient_data)
                
                # Display results
                st.markdown("## Multi-Disease Risk Assessment")
                
                # Risk cards
                risk_scores = risk_summary['risk_scores']
                risk_categories = risk_summary['risk_categories']
                
                cols = st.columns(3)
                for i, (disease, score) in enumerate(risk_scores.items()):
                    with cols[i]:
                        category = risk_categories[disease]
                        risk_class = f"risk-{category.lower()}"
                        
                        st.markdown(f"""
                        <div class="risk-card {risk_class}">
                            <h3>{disease.title()}</h3>
                            <h2>{score:.1%}</h2>
                            <p>{category} Risk</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detailed analysis
                st.markdown("## Detailed Analysis")
                
                # Temporal projections
                st.markdown("### Temporal Risk Projections")
                
                # Get temporal projections for diabetes (primary risk)
                primary_disease = risk_summary['primary_risk'][0] if risk_summary['primary_risk'] else 'diabetes'
                
                try:
                    projections = {}
                    for months in [6, 12]:
                        result = projector.project_forward(primary_disease, patient_data, months)
                        projections[f'{months}_months'] = result
                    
                    projections['current'] = risk_scores[primary_disease]
                    
                    # Intervention simulations
                    simulator = InterventionSimulator()
                    interventions = simulator.simulate_interventions(primary_disease, patient_data)
                    
                    # Create timeline chart
                    fig = create_timeline_chart(projections, interventions)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Intervention simulator
                    st.markdown("### Intervention Simulator")
                    st.write("See how lifestyle changes affect your 6-month risk:")
                    
                    # Interactive sliders
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**BMI Change**")
                        bmi_change = st.slider("BMI reduction", -10.0, 5.0, 0.0, 0.5, 
                                             help="Negative values = weight loss")
                        st.markdown("**Glucose Change (mg/dL)**")
                        glucose_change = st.slider("Glucose reduction", -50, 20, 0, 5,
                                                 help="Negative values = improvement")
                    
                    with col2:
                        # Calculate new risk
                        intervention = {
                            'bmi_reduction': -bmi_change if bmi_change < 0 else 0,
                            'exercise_program': bmi_change < 0
                        }
                        
                        new_result = projector.project_forward(
                            primary_disease, patient_data, 6, intervention
                        )
                        
                        current_risk = risk_scores[primary_disease]
                        risk_reduction = current_risk - new_result['mean_risk']
                        
                        st.metric(
                            "Projected 6-Month Risk",
                            f"{new_result['mean_risk']:.1%}",
                            delta=f"{risk_reduction:+.1%}"
                        )
                        
                        st.success(f"Risk reduction: {risk_reduction:.1%}" if risk_reduction > 0 else "No improvement")
                
                except Exception as e:
                    st.error(f"Error in temporal projection: {e}")
                
                # Explainability
                st.markdown("### Explainability Trinity")
                
                try:
                    explanation = explainer.get_comprehensive_explanation(primary_disease, patient_data)
                    
                    if explanation:
                        # Natural language explanation
                        st.markdown("#### Patient-Friendly Explanation")
                        st.info(explanation['natural_language_explanations'])
                        
                        # SHAP waterfall (simplified)
                        if explanation['shap_explanation']:
                            st.markdown("#### Key Risk Factors")
                            feature_importance = explanation['shap_explanation']['feature_importance'][:5]
                            
                            for factor in feature_importance:
                                direction = "increases" if factor['contribution'] > 0 else "decreases"
                                st.write(f"**{factor['feature']}**: {factor['value']:.1f} ({direction} risk)")
                        
                        # Counterfactual
                        if explanation['counterfactual_explanation']:
                            st.markdown("#### Path to Low Risk")
                            cf = explanation['counterfactual_explanation']
                            if 'summary' in cf:
                                st.info(cf['summary'])
                
                except Exception as e:
                    st.error(f"Error in explainability: {e}")
                
                # Generate care pathway
                st.markdown("## Personalized Care Pathway")
                
                if st.button("Generate 12-Week Care Pathway", type="secondary"):
                    with st.spinner("Generating personalized care pathway..."):
                        try:
                            # This would require GROQ_API_KEY
                            pathway_result = agent.analyze_patient(patient_data)
                            
                            if 'error' in pathway_result:
                                st.warning("Care pathway generation requires GROQ_API_KEY environment variable.")
                            else:
                                pathway = pathway_result.get('care_pathway', {})
                                
                                # Display pathway sections
                                if 'risk_summary' in pathway:
                                    st.markdown("### Risk Summary")
                                    st.write(pathway['risk_summary'])
                                
                                if 'phased_intervention' in pathway:
                                    st.markdown("### 12-Week Intervention Plan")
                                    st.write(pathway['phased_intervention'])
                                
                                if 'recommendations' in pathway:
                                    st.markdown("### Lifestyle Recommendations")
                                    for rec in pathway['recommendations']:
                                        st.write(f"- {rec}")
                        
                        except Exception as e:
                            st.error(f"Error generating care pathway: {e}")
    
    with batch_tab:
        st.markdown("### Batch Upload")
        st.write("Upload a CSV file with patient data for batch analysis.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                if st.button("Analyze Batch"):
                    with st.spinner("Processing batch data..."):
                        results = []
                        
                        for _, row in df.iterrows():
                            # Convert row to patient data dict
                            patient_batch_data = dict(row)
                            
                            # Get risk summary
                            batch_summary = agent.get_risk_summary(patient_batch_data)
                            
                            result = {
                                'patient_id': row.get('patient_id', f"Patient_{len(results)+1}"),
                                'diabetes_risk': batch_summary['risk_scores'].get('diabetes', 0),
                                'heart_risk': batch_summary['risk_scores'].get('heart', 0),
                                'kidney_risk': batch_summary['risk_scores'].get('kidney', 0),
                                'primary_risk': batch_summary['primary_risk'][0] if batch_summary['primary_risk'] else 'unknown'
                            }
                            results.append(result)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        
                        st.write("Batch Results:")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name=f"healthguard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with about_tab:
        st.markdown("## About HealthGuard AI")
        
        st.markdown("""
        ### System Overview
        HealthGuard AI is an advanced multi-disease risk assessment system that combines:
        
        - **Ensemble Machine Learning**: 9 models across 3 diseases (Diabetes, Heart Disease, Kidney Disease)
        - **Temporal Risk Projection**: Predicts future risk with uncertainty quantification
        - **Intervention Simulation**: Shows how lifestyle changes affect risk trajectories
        - **Explainability Trinity**: SHAP technical analysis + Natural language explanations + Counterfactual analysis
        - **Agentic AI**: LangGraph-powered personalized care pathway generation
        - **Medical Knowledge Graph**: Evidence-based recommendations with RAG
        
        ### Model Performance
        """)
        
        if metrics:
            for disease, disease_metrics in metrics.items():
                st.markdown(f"**{disease.title()}**:")
                st.write(f"- Best Model: {disease_metrics.get('best_model', 'N/A')}")
                st.write(f"- Accuracy: {disease_metrics.get('accuracy', 0):.1%}")
                st.write(f"- ROC-AUC: {disease_metrics.get('roc_auc', 0):.3f}")
                st.write("")
        
        st.markdown("""
        ### Technical Stack
        - **ML**: scikit-learn, XGBoost, SHAP
        - **Temporal**: Custom projection with uncertainty quantification
        - **LLM**: Groq (Llama 3.1) for care pathway generation
        - **Knowledge**: FAISS + Sentence Transformers for RAG
        - **Agent**: LangGraph for workflow orchestration
        - **UI**: Streamlit with Plotly visualizations
        
        ### Medical Disclaimer
        This system is for educational and informational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always seek the advice of qualified health providers with any questions you may have.
        """)

if __name__ == "__main__":
    main()
