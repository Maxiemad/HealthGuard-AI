from typing import TypedDict, Literal, List, Dict, Optional

class AgentState(TypedDict):
    """State schema for the HealthGuard AI agent workflow"""
    
    # Input data
    patient_data: Dict
    risk_scores: Dict[str, float]  # Disease -> risk score
    risk_categories: Dict[str, str]  # Disease -> category
    feature_importance: Dict[str, List[Dict]]  # Disease -> feature importance list
    
    # Temporal projections
    temporal_projections: Dict[str, Dict]  # Disease -> projection data
    intervention_effects: Dict[str, Dict]  # Disease -> intervention results
    
    # Explainability
    shap_explanations: Dict[str, Dict]
    natural_language_explanations: Dict[str, str]
    counterfactual_explanations: Dict[str, Dict]
    
    # Processing state
    current_step: Literal[
        "risk_analysis", 
        "temporal_projection", 
        "explainability_analysis",
        "knowledge_retrieval", 
        "care_pathway_generation", 
        "complete"
    ]
    
    # Medical knowledge
    retrieved_knowledge: str
    evidence_chains: List[str]
    
    # Output
    care_pathway: Dict
    health_report: Dict
    recommendations: List[str]
    follow_up_schedule: Dict
    
    # Error handling
    error: Optional[str]
    warnings: List[str]
