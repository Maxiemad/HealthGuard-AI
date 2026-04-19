# HealthGuard AI: Multi-Disease Risk Intelligence System

> **Advanced healthcare risk assessment with temporal projections, intervention simulation, and AI-powered care pathways**

## Overview

HealthGuard AI represents a breakthrough in preventive healthcare analytics, combining ensemble machine learning, temporal risk projection, and agentic AI to provide comprehensive multi-disease risk assessment and personalized care planning.

### Key Innovations

- **Multi-Disease Ensemble**: 9 models across 3 diseases (Diabetes, Heart Disease, Kidney Disease)
- **Temporal Risk Projection**: Predict future risk trajectories with uncertainty quantification
- **Intervention Simulator**: Interactive "what-if" scenarios showing how lifestyle changes affect risk
- **Explainability Trinity**: SHAP technical analysis + Natural language explanations + Counterfactual analysis
- **Agentic Care Pathways**: LangGraph-powered 12-week personalized intervention plans
- **Medical Knowledge Graph**: Evidence-based recommendations with RAG system

## Architecture

```
Patient Input
    |
    v
Multi-Disease Ensemble Models
    |
    v
Temporal Risk Projector + Intervention Simulator
    |
    v
Explainability Trinity (SHAP + LLM + Counterfactual)
    |
    v
LangGraph Agent + Medical Knowledge Graph
    |
    v
Personalized Care Pathway + PDF Report
```

## Features

### 1. Multi-Disease Risk Assessment
- **Diabetes**: Logistic Regression ensemble with feature engineering
- **Heart Disease**: Ensemble model with cardiovascular risk factors
- **Kidney Disease**: Predictive model with renal function markers

### 2. Temporal Risk Projection
- 6-month and 1-year risk projections
- Uncertainty quantification with confidence intervals
- Natural disease progression modeling

### 3. Intervention Simulator
- Interactive sliders for BMI and glucose changes
- Real-time risk recalculation
- Multiple intervention scenarios (Lifestyle, Aggressive, Exercise-only)

### 4. Explainability Trinity
- **Level 1**: SHAP waterfall plots for technical interpretation
- **Level 2**: LLM-generated patient-friendly explanations
- **Level 3**: Counterfactual analysis ("What would it take to be low risk?")

### 5. AI-Powered Care Pathways
- 12-week phased intervention plans
- Evidence-based recommendations
- Follow-up scheduling
- Progress monitoring metrics

### 6. Comprehensive UI
- Real-time risk visualization
- Interactive timeline charts
- Batch processing capabilities
- PDF report generation

## Technical Stack

### Machine Learning
- **Ensemble Models**: scikit-learn, XGBoost
- **Explainability**: SHAP
- **Temporal Modeling**: Custom projection with bootstrap uncertainty

### AI & NLP
- **LLM**: Groq API (Llama 3.1-8b-instant)
- **Agent Framework**: LangGraph
- **Knowledge Retrieval**: FAISS + Sentence Transformers

### Frontend
- **UI Framework**: Streamlit
- **Visualization**: Plotly
- **PDF Generation**: ReportLab

### Data Processing
- **Data Manipulation**: pandas, numpy
- **Feature Engineering**: Custom pipelines
- **Model Persistence**: joblib

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/HealthGuard-AI.git
   cd HealthGuard-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train ensemble models**
   ```bash
   python ensemble_training.py
   ```

4. **Build knowledge base index**
   ```bash
   python rag/build_index.py
   ```

5. **Run the application**
   ```bash
   streamlit run healthguard_app.py
   ```

### Environment Variables (Optional)
For full AI-powered care pathway generation:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

## Usage

### Individual Assessment
1. Enter patient demographics and clinical measurements
2. Click "Comprehensive Risk Analysis"
3. Review multi-disease risk assessment
4. Explore temporal projections and intervention scenarios
5. Generate personalized care pathway
6. Download PDF report

### Batch Processing
1. Navigate to "Batch Processing" tab
2. Upload CSV file with patient data
3. Click "Analyze Batch"
4. Review results and download analysis

## Model Performance

### Diabetes Prediction
- **Best Model**: Logistic Regression (with class balancing)
- **ROC-AUC**: 0.826 ± 0.022
- **Accuracy**: 74.0%
- **Features**: 8 clinical measurements + engineered categories

### Heart Disease Prediction
- **Best Model**: Logistic Regression
- **ROC-AUC**: 0.750 ± 0.060
- **Accuracy**: 66.7%
- **Features**: 14 cardiovascular risk factors

### Kidney Disease Prediction
- **Best Model**: Logistic Regression
- **ROC-AUC**: 1.000 (perfect separation on dataset)
- **Accuracy**: 100.0%
- **Features**: 24 renal function markers

## Project Structure

```
HealthGuard-AI/
|
|--- data/                    # Dataset files
|    |--- diabetes.csv
|    |--- heart/
|    |--- kidney/
|
|--- models/                  # Trained models and artifacts
|    |--- *_model.pkl
|    |--- *_scaler.pkl
|    |--- metrics.json
|
|--- agent/                   # LangGraph agent system
|    |--- state.py
|    |--- nodes.py
|    |--- graph.py
|
|--- rag/                     # Knowledge retrieval system
|    |--- knowledge_base/
|    |--- build_index.py
|    |--- retriever.py
|
|--- utils/                   # Utility functions
|    |--- pdf_export.py
|
|--- healthguard_app.py       # Main application
|--- ensemble_training.py     # Model training pipeline
|--- temporal_projector.py   # Risk projection system
|--- explainability.py       # Explainability trinity
```

## API Reference

### Core Classes

#### `MultiDiseaseEnsemble`
```python
ensemble = MultiDiseaseEnsemble()
ensemble.train_all_models()
ensemble.save_artifacts()
```

#### `TemporalRiskProjector`
```python
projector = TemporalRiskProjector()
result = projector.project_forward('diabetes', patient_data, 6)
```

#### `ExplainabilityTrinity`
```python
explainer = ExplainabilityTrinity()
explanation = explainer.get_comprehensive_explanation('diabetes', patient_data)
```

#### `HealthGuardAgent`
```python
agent = get_agent()
analysis = agent.analyze_patient(patient_data)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Medical Disclaimer

This system is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions you may have regarding a medical condition.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{healthguard_ai,
  title={HealthGuard AI: Multi-Disease Risk Intelligence System},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-username/HealthGuard-AI}
}
```

## Acknowledgments

- Pima Indians Diabetes Dataset (National Institute of Diabetes and Digestive and Kidney Diseases)
- Cleveland Heart Disease Dataset (UCI Machine Learning Repository)
- Chronic Kidney Disease Dataset (various sources)
- Groq for providing free API access for LLM inference
- Streamlit for the excellent web framework
- Open-source community for the amazing ML and AI tools
