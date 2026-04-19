from langgraph.graph import StateGraph, END
from typing import Dict, Any

from agent.state import AgentState
from agent.nodes import HealthGuardAgentNodes

def create_health_agent():
    """Create the HealthGuard AI agent workflow"""
    
    # Initialize node functions
    nodes = HealthGuardAgentNodes()
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("risk_analyzer", nodes.risk_analyzer_node)
    workflow.add_node("temporal_projection", nodes.temporal_projection_node)
    workflow.add_node("explainability_analysis", nodes.explainability_node)
    workflow.add_node("knowledge_retrieval", nodes.knowledge_retrieval_node)
    workflow.add_node("care_pathway_generator", nodes.care_pathway_generator_node)
    
    # Define workflow edges
    workflow.set_entry_point("risk_analyzer")
    
    workflow.add_edge("risk_analyzer", "temporal_projection")
    workflow.add_edge("temporal_projection", "explainability_analysis")
    workflow.add_edge("explainability_analysis", "knowledge_retrieval")
    workflow.add_edge("knowledge_retrieval", "care_pathway_generator")
    workflow.add_edge("care_pathway_generator", END)
    
    return workflow.compile()

class HealthGuardAgent:
    """Main agent interface for HealthGuard AI"""
    
    def __init__(self):
        self.workflow = create_health_agent()
    
    def analyze_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete patient analysis workflow
        
        Args:
            patient_data: Dictionary containing patient features
            
        Returns:
            Complete analysis results including care pathway
        """
        
        # Import ensemble models to get risk scores
        import joblib
        
        # Load models
        models = {}
        scalers = {}
        feature_names = joblib.load("models/feature_names.pkl")
        
        diseases = ['diabetes', 'heart', 'kidney']
        for disease in diseases:
            try:
                models[disease] = joblib.load(f"models/{disease}_model.pkl")
                scalers[disease] = joblib.load(f"models/{disease}_scaler.pkl")
            except:
                print(f"Warning: Could not load {disease} model")
        
        # Calculate risk scores
        risk_scores = {}
        for disease in diseases:
            if disease in models:
                try:
                    # Prepare features
                    expected_features = feature_names[disease]
                    patient_features = {}
                    for feature in expected_features:
                        patient_features[feature] = patient_data.get(feature, 0)
                    
                    # Create DataFrame and predict
                    import pandas as pd
                    features_df = pd.DataFrame([patient_features])
                    scaled_features = scalers[disease].transform(features_df)
                    risk_score = models[disease].predict_proba(scaled_features)[0][1]
                    risk_scores[disease] = risk_score
                    
                except Exception as e:
                    print(f"Error predicting {disease}: {e}")
                    risk_scores[disease] = 0.5  # Default
        
        # Initialize state
        initial_state = AgentState(
            patient_data=patient_data,
            risk_scores=risk_scores,
            risk_categories={},
            feature_importance={},
            temporal_projections={},
            intervention_effects={},
            shap_explanations={},
            natural_language_explanations={},
            counterfactual_explanations={},
            current_step="risk_analysis",
            retrieved_knowledge="",
            evidence_chains=[],
            care_pathway={},
            health_report={},
            recommendations=[],
            follow_up_schedule={},
            error=None,
            warnings=[]
        )
        
        # Run workflow
        try:
            result = self.workflow.invoke(initial_state)
            return result
        except Exception as e:
            return {
                'error': f'Workflow execution failed: {str(e)}',
                'patient_data': patient_data,
                'risk_scores': risk_scores
            }
    
    def get_risk_summary(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Quick risk summary without full pathway generation"""
        
        # Import ensemble models
        import joblib
        import pandas as pd
        
        # Load models
        models = {}
        scalers = {}
        feature_names = joblib.load("models/feature_names.pkl")
        
        diseases = ['diabetes', 'heart', 'kidney']
        for disease in diseases:
            try:
                models[disease] = joblib.load(f"models/{disease}_model.pkl")
                scalers[disease] = joblib.load(f"models/{disease}_scaler.pkl")
            except:
                continue
        
        # Calculate risk scores
        risk_scores = {}
        risk_categories = {}
        
        for disease in diseases:
            if disease in models:
                try:
                    # Prepare features
                    expected_features = feature_names[disease]
                    patient_features = {}
                    for feature in expected_features:
                        patient_features[feature] = patient_data.get(feature, 0)
                    
                    # Create DataFrame and predict
                    features_df = pd.DataFrame([patient_features])
                    scaled_features = scalers[disease].transform(features_df)
                    risk_score = models[disease].predict_proba(scaled_features)[0][1]
                    risk_scores[disease] = risk_score
                    
                    # Categorize risk
                    if risk_score < 0.4:
                        risk_categories[disease] = "Low"
                    elif risk_score < 0.7:
                        risk_categories[disease] = "Moderate"
                    else:
                        risk_categories[disease] = "High"
                        
                except Exception as e:
                    print(f"Error predicting {disease}: {e}")
                    risk_scores[disease] = 0.5
                    risk_categories[disease] = "Unknown"
        
        return {
            'risk_scores': risk_scores,
            'risk_categories': risk_categories,
            'primary_risk': max(risk_scores.items(), key=lambda x: x[1]) if risk_scores else None
        }

# Global agent instance
_agent = None

def get_agent():
    """Get or create the global agent instance"""
    global _agent
    if _agent is None:
        _agent = HealthGuardAgent()
    return _agent

if __name__ == "__main__":
    # Test the agent
    agent = get_agent()
    
    # Example patient data
    patient_data = {
        'Pregnancies': 2,
        'Glucose': 145,
        'BloodPressure': 80,
        'SkinThickness': 20,
        'Insulin': 85,
        'BMI': 32.5,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 45
    }
    
    # Quick risk summary
    summary = agent.get_risk_summary(patient_data)
    print("Risk Summary:")
    for disease, score in summary['risk_scores'].items():
        category = summary['risk_categories'][disease]
        print(f"  {disease}: {score:.1%} ({category})")
    
    print(f"\nPrimary Risk: {summary['primary_risk']}")
