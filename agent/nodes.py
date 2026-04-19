from groq import Groq
import os
import json
from typing import Dict, List
from datetime import datetime, timedelta
import networkx as nx

from agent.state import AgentState
from rag.retriever import get_retriever
from temporal_projector import TemporalRiskProjector
from explainability import ExplainabilityTrinity

class HealthGuardAgentNodes:
    def __init__(self):
        """Initialize agent nodes with necessary components"""
        self.llm_client = None
        if os.getenv("GROQ_API_KEY"):
            self.llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        self.retriever = get_retriever()
        self.temporal_projector = TemporalRiskProjector()
        self.explainability = ExplainabilityTrinity()
        
        # Build medical knowledge graph
        self.knowledge_graph = self._build_knowledge_graph()
    
    def _build_knowledge_graph(self):
        """Build a simple medical knowledge graph for evidence chains"""
        G = nx.DiGraph()
        
        # Add disease-risk factor relationships
        risk_factors = {
            'High BMI': ['diabetes', 'heart_disease', 'kidney_disease'],
            'High Glucose': ['diabetes', 'heart_disease', 'kidney_disease'],
            'High Blood Pressure': ['heart_disease', 'kidney_disease'],
            'High Cholesterol': ['heart_disease'],
            'Smoking': ['heart_disease', 'kidney_disease'],
            'Age': ['diabetes', 'heart_disease', 'kidney_disease'],
            'Family History': ['diabetes', 'heart_disease', 'kidney_disease']
        }
        
        for factor, diseases in risk_factors.items():
            for disease in diseases:
                G.add_edge(factor, disease, relationship='increases_risk')
        
        # Add intervention relationships
        interventions = {
            'Exercise': ['High BMI', 'High Blood Pressure', 'High Glucose'],
            'Diet': ['High BMI', 'High Cholesterol', 'High Glucose'],
            'Medication': ['High Blood Pressure', 'High Glucose', 'High Cholesterol'],
            'Weight Loss': ['High BMI', 'High Blood Pressure', 'High Glucose']
        }
        
        for intervention, factors in interventions.items():
            for factor in factors:
                G.add_edge(intervention, factor, relationship='reduces')
        
        return G
    
    def risk_analyzer_node(self, state: AgentState) -> AgentState:
        """Analyze risk scores and identify primary/secondary risks"""
        
        risk_scores = state['risk_scores']
        
        # Sort risks by score
        sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Identify primary and secondary risks
        primary_risk = sorted_risks[0] if sorted_risks else None
        secondary_risks = sorted_risks[1:3] if len(sorted_risks) > 1 else []
        
        # Risk categorization
        risk_categories = {}
        for disease, score in risk_scores.items():
            if score < 0.4:
                risk_categories[disease] = "Low"
            elif score < 0.7:
                risk_categories[disease] = "Moderate"
            else:
                risk_categories[disease] = "High"
        
        # Add warnings for high risks
        warnings = []
        for disease, score in risk_scores.items():
            if score >= 0.7:
                warnings.append(f"High {disease} risk ({score:.1%}) requires immediate attention")
        
        state['risk_categories'] = risk_categories
        state['warnings'] = warnings
        state['current_step'] = 'temporal_projection'
        
        return state
    
    def temporal_projection_node(self, state: AgentState) -> AgentState:
        """Generate temporal risk projections and intervention effects"""
        
        patient_data = state['patient_data']
        risk_scores = state['risk_scores']
        
        temporal_projections = {}
        intervention_effects = {}
        
        for disease in risk_scores.keys():
            try:
                # 6-month and 1-year projections
                projection_6mo = self.temporal_projector.project_forward(
                    disease, patient_data, 6
                )
                projection_1yr = self.temporal_projector.project_forward(
                    disease, patient_data, 12
                )
                
                temporal_projections[disease] = {
                    'current': risk_scores[disease],
                    '6_months': projection_6mo,
                    '1_year': projection_1yr
                }
                
                # Intervention simulations
                interventions = {
                    'Lifestyle Change': {
                        'bmi_reduction': 2.0,
                        'exercise_program': True
                    },
                    'Aggressive Lifestyle': {
                        'bmi_reduction': 4.0,
                        'exercise_program': True
                    }
                }
                
                intervention_effects[disease] = {}
                for intervention_name, params in interventions.items():
                    future_result = self.temporal_projector.project_forward(
                        disease, patient_data, 6, params
                    )
                    intervention_effects[disease][intervention_name] = {
                        'risk_reduction': risk_scores[disease] - future_result['mean_risk'],
                        'new_risk': future_result['mean_risk'],
                        'confidence_interval': (
                            future_result['ci_lower'], 
                            future_result['ci_upper']
                        )
                    }
                
            except Exception as e:
                print(f"Error in temporal projection for {disease}: {e}")
                temporal_projections[disease] = {'error': str(e)}
                intervention_effects[disease] = {'error': str(e)}
        
        state['temporal_projections'] = temporal_projections
        state['intervention_effects'] = intervention_effects
        state['current_step'] = 'explainability_analysis'
        
        return state
    
    def explainability_node(self, state: AgentState) -> AgentState:
        """Generate comprehensive explanations for all diseases"""
        
        patient_data = state['patient_data']
        risk_scores = state['risk_scores']
        
        shap_explanations = {}
        natural_language_explanations = {}
        counterfactual_explanations = {}
        feature_importance = {}
        
        for disease in risk_scores.keys():
            try:
                # Get comprehensive explanation
                explanation = self.explainability.get_comprehensive_explanation(
                    disease, patient_data
                )
                
                if explanation:
                    shap_explanations[disease] = explanation['shap_explanation']
                    natural_language_explanations[disease] = explanation['natural_language_explanation']
                    counterfactual_explanations[disease] = explanation['counterfactual_explanation']
                    
                    if explanation['shap_explanation']:
                        feature_importance[disease] = explanation['shap_explanation']['feature_importance']
                
            except Exception as e:
                print(f"Error in explainability for {disease}: {e}")
                shap_explanations[disease] = {'error': str(e)}
                natural_language_explanations[disease] = f"Error: {str(e)}"
                counterfactual_explanations[disease] = {'error': str(e)}
        
        state['shap_explanations'] = shap_explanations
        state['natural_language_explanations'] = natural_language_explanations
        state['counterfactual_explanations'] = counterfactual_explanations
        state['feature_importance'] = feature_importance
        state['current_step'] = 'knowledge_retrieval'
        
        return state
    
    def knowledge_retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant medical knowledge and build evidence chains"""
        
        risk_scores = state['risk_scores']
        feature_importance = state['feature_importance']
        
        # Build queries for each disease
        knowledge_queries = {}
        for disease, score in risk_scores.items():
            if score > 0.4:  # Only retrieve for moderate/high risk
                # Get top risk factors
                top_factors = []
                if disease in feature_importance and feature_importance[disease]:
                    top_factors = [f['feature'] for f in feature_importance[disease][:3]]
                
                # Create disease-specific query
                query = f"{disease} prevention management"
                if top_factors:
                    query += f" {' '.join(top_factors[:2])}"
                
                knowledge_queries[disease] = query
        
        # Retrieve knowledge
        retrieved_knowledge = {}
        for disease, query in knowledge_queries.items():
            try:
                context = self.retriever.retrieve(query, top_k=3)
                retrieved_knowledge[disease] = context
            except Exception as e:
                retrieved_knowledge[disease] = f"Error retrieving knowledge: {str(e)}"
        
        # Build evidence chains using knowledge graph
        evidence_chains = []
        for disease in risk_scores.keys():
            if disease in feature_importance and feature_importance[disease]:
                top_factor = feature_importance[disease][0]['feature']
                
                # Find path from risk factor to disease
                try:
                    if self.knowledge_graph.has_node(top_factor):
                        paths = list(nx.all_simple_paths(
                            self.knowledge_graph, 
                            top_factor, 
                            disease, 
                            cutoff=3
                        ))
                        if paths:
                            chain = " -> ".join(paths[0])
                            evidence_chains.append(f"{disease}: {chain}")
                except:
                    pass
        
        state['retrieved_knowledge'] = json.dumps(retrieved_knowledge, indent=2)
        state['evidence_chains'] = evidence_chains
        state['current_step'] = 'care_pathway_generation'
        
        return state
    
    def care_pathway_generator_node(self, state: AgentState) -> AgentState:
        """Generate phased care pathway using LLM"""
        
        if not self.llm_client:
            state['care_pathway'] = {'error': 'LLM not available - please set GROQ_API_KEY'}
            state['current_step'] = 'complete'
            return state
        
        # Prepare comprehensive prompt
        prompt = self._create_care_pathway_prompt(state)
        
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            pathway_text = response.choices[0].message.content
            
            # Parse the structured response
            care_pathway = self._parse_care_pathway(pathway_text)
            
            # Add follow-up schedule
            care_pathway['follow_up_schedule'] = self._generate_follow_up_schedule(state)
            
            state['care_pathway'] = care_pathway
            state['health_report'] = self._generate_health_report(state)
            state['recommendations'] = care_pathway.get('recommendations', [])
            
        except Exception as e:
            state['care_pathway'] = {'error': f'Error generating care pathway: {str(e)}'}
        
        state['current_step'] = 'complete'
        return state
    
    def _create_care_pathway_prompt(self, state: AgentState) -> str:
        """Create comprehensive prompt for care pathway generation"""
        
        risk_scores = state['risk_scores']
        risk_categories = state['risk_categories']
        temporal_projections = state['temporal_projections']
        intervention_effects = state['intervention_effects']
        natural_language_explanations = state['natural_language_explanations']
        retrieved_knowledge = state['retrieved_knowledge']
        evidence_chains = state['evidence_chains']
        
        prompt = f"""You are an expert medical care coordinator creating a personalized 12-week intervention plan.

PATIENT RISK ASSESSMENT:
"""
        
        for disease, score in risk_scores.items():
            category = risk_categories.get(disease, 'Unknown')
            explanation = natural_language_explanations.get(disease, 'No explanation available')
            
            prompt += f"""
{disease.title()}: {score:.1%} risk ({category})
Current explanation: {explanation}
"""
            
            if disease in temporal_projections and '6_months' in temporal_projections[disease]:
                proj_6mo = temporal_projections[disease]['6_months']
                if 'mean_risk' in proj_6mo:
                    prompt += f"6-month projection: {proj_6mo['mean_risk']:.1%} risk\n"
            
            if disease in intervention_effects:
                effects = intervention_effects[disease]
                if 'Lifestyle Change' in effects:
                    effect = effects['Lifestyle Change']
                    if 'risk_reduction' in effect:
                        prompt += f"Lifestyle change potential: {effect['risk_reduction']:.1%} risk reduction\n"
        
        prompt += f"""
MEDICAL KNOWLEDGE BASE:
{retrieved_knowledge}

EVIDENCE CHAINS:
{'; '.join(evidence_chains) if evidence_chains else 'No evidence chains available'}

Generate a comprehensive 12-week phased care pathway with these exact sections:

1. RISK SUMMARY (2-3 sentences)
2. PRIMARY FOCUS (main disease and target risk reduction)
3. PHASED INTERVENTION PLAN:
   - Week 1-2: [Focus area, specific actions, success metrics, escalation triggers]
   - Week 3-4: [Next focus, actions, metrics, triggers]
   - Week 5-8: [Consolidation phase, actions, metrics]
   - Week 9-12: [Maintenance phase, actions, reassessment]
4. LIFESTYLE RECOMMENDATIONS (5 specific, evidence-based actions)
5. MONITORING PLAN (what to track, frequency, warning signs)
6. EXPECTED OUTCOMES (quantitative projections with confidence intervals)
7. MEDICAL DISCLAIMER

Be specific, actionable, and evidence-based. Include exact measurements and timelines.
"""
        
        return prompt
    
    def _parse_care_pathway(self, pathway_text: str) -> Dict:
        """Parse LLM response into structured care pathway"""
        sections = {}
        current_section = None
        content_lines = []
        
        for line in pathway_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if 'RISK SUMMARY' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(content_lines)
                current_section = 'risk_summary'
                content_lines = []
            elif 'PRIMARY FOCUS' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(content_lines)
                current_section = 'primary_focus'
                content_lines = []
            elif 'PHASED INTERVENTION' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(content_lines)
                current_section = 'phased_intervention'
                content_lines = []
            elif 'LIFESTYLE RECOMMENDATIONS' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(content_lines)
                current_section = 'lifestyle_recommendations'
                content_lines = []
            elif 'MONITORING PLAN' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(content_lines)
                current_section = 'monitoring_plan'
                content_lines = []
            elif 'EXPECTED OUTCOMES' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(content_lines)
                current_section = 'expected_outcomes'
                content_lines = []
            elif 'MEDICAL DISCLAIMER' in line.upper():
                if current_section:
                    sections[current_section] = '\n'.join(content_lines)
                current_section = 'medical_disclaimer'
                content_lines = []
            else:
                if current_section:
                    content_lines.append(line)
        
        # Add last section
        if current_section and content_lines:
            sections[current_section] = '\n'.join(content_lines)
        
        # Extract recommendations from lifestyle section
        recommendations = []
        if 'lifestyle_recommendations' in sections:
            lines = sections['lifestyle_recommendations'].split('\n')
            for line in lines:
                if line.strip() and (line.startswith('-') or line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.')):
                    recommendations.append(line.strip().lstrip('-123456789. ').strip())
        
        sections['recommendations'] = recommendations
        
        return sections
    
    def _generate_follow_up_schedule(self, state: AgentState) -> Dict:
        """Generate follow-up schedule based on risk levels"""
        
        risk_scores = state['risk_scores']
        max_risk = max(risk_scores.values()) if risk_scores else 0
        
        if max_risk >= 0.7:
            return {
                'initial_follow_up': '2 weeks',
                'subsequent_visits': 'Every 4 weeks',
                'lab_monitoring': 'Every 8 weeks',
                'specialist_referral': 'Consider immediate referral'
            }
        elif max_risk >= 0.4:
            return {
                'initial_follow_up': '4 weeks',
                'subsequent_visits': 'Every 8 weeks',
                'lab_monitoring': 'Every 12 weeks',
                'specialist_referral': 'Consider if no improvement'
            }
        else:
            return {
                'initial_follow_up': '8 weeks',
                'subsequent_visits': 'Every 12 weeks',
                'lab_monitoring': 'Every 6 months',
                'specialist_referral': 'As needed'
            }
    
    def _generate_health_report(self, state: AgentState) -> Dict:
        """Generate comprehensive health report summary"""
        
        return {
            'generated_date': datetime.now().isoformat(),
            'risk_summary': state['risk_categories'],
            'temporal_projections': state['temporal_projections'],
            'intervention_potential': state['intervention_effects'],
            'evidence_based': True,
            'personalized': True
        }
