from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io
import base64

def generate_health_report_pdf(patient_data, risk_scores, risk_categories, 
                               temporal_projections=None, care_pathway=None,
                               shap_explanation=None, natural_language_explanation=None):
    """Generate comprehensive health report PDF"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=12
    )
    
    # Title page
    story.append(Paragraph("HealthGuard AI - Health Risk Assessment Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report info
    report_info = [
        ['Report Date:', datetime.now().strftime('%B %d, %Y')],
        ['Patient ID:', f"Patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"],
        ['Assessment Type:', 'Multi-Disease Risk Analysis']
    ]
    
    info_table = Table(report_info, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 30))
    
    # Risk Summary Section
    story.append(Paragraph("Risk Assessment Summary", heading_style))
    
    # Create risk table
    risk_data = [['Disease', 'Risk Score', 'Risk Category', 'Status']]
    
    for disease, score in risk_scores.items():
        category = risk_categories.get(disease, 'Unknown')
        
        # Determine status and color
        if score < 0.4:
            status = 'Low Risk'
            status_color = colors.green
        elif score < 0.7:
            status = 'Moderate Risk'
            status_color = colors.orange
        else:
            status = 'High Risk'
            status_color = colors.red
        
        risk_data.append([disease.title(), f'{score:.1%}', category, status])
    
    risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10)
    ]))
    
    # Color code risk levels
    for i in range(1, len(risk_data)):
        score = float(risk_data[i][1].rstrip('%')) / 100
        if score >= 0.7:
            risk_table.setStyle(TableStyle([
                ('TEXTCOLOR', (3, i), (3, i), colors.red)
            ]))
        elif score >= 0.4:
            risk_table.setStyle(TableStyle([
                ('TEXTCOLOR', (3, i), (3, i), colors.orange)
            ]))
        else:
            risk_table.setStyle(TableStyle([
                ('TEXTCOLOR', (3, i), (3, i), colors.green)
            ]))
    
    story.append(risk_table)
    story.append(Spacer(1, 20))
    
    # Patient Data Summary
    story.append(Paragraph("Patient Profile", heading_style))
    
    patient_summary = []
    for key, value in patient_data.items():
        if not key.startswith(('BMI_Category_', 'Glucose_Category_', 'Age_Group_')):
            patient_summary.append([key.replace('_', ' ').title(), str(value)])
    
    patient_table = Table(patient_summary, colWidths=[2.5*inch, 1.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Natural Language Explanation
    if natural_language_explanation:
        story.append(Paragraph("Risk Explanation", heading_style))
        story.append(Paragraph(natural_language_explanation, styles['BodyText']))
        story.append(Spacer(1, 20))
    
    # Temporal Projections
    if temporal_projections:
        story.append(Paragraph("Risk Projections", heading_style))
        
        projection_data = [['Time Point', 'Risk Score', 'Confidence Interval']]
        
        # Current
        current_risk = temporal_projections.get('current', 0)
        projection_data.append(['Current', f'{current_risk:.1%}', 'N/A'])
        
        # 6 months
        proj_6mo = temporal_projections.get('6_months', {})
        if proj_6mo:
            risk_6mo = proj_6mo.get('mean_risk', 0)
            ci_lower = proj_6mo.get('ci_lower', 0)
            ci_upper = proj_6mo.get('ci_upper', 0)
            projection_data.append(['6 Months', f'{risk_6mo:.1%}', f'{ci_lower:.1%} - {ci_upper:.1%}'])
        
        # 1 year
        proj_1yr = temporal_projections.get('1_year', {})
        if proj_1yr:
            risk_1yr = proj_1yr.get('mean_risk', 0)
            ci_lower = proj_1yr.get('ci_lower', 0)
            ci_upper = proj_1yr.get('ci_upper', 0)
            projection_data.append(['1 Year', f'{risk_1yr:.1%}', f'{ci_lower:.1%} - {ci_upper:.1%}'])
        
        projection_table = Table(projection_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        projection_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10)
        ]))
        
        story.append(projection_table)
        story.append(Spacer(1, 20))
    
    # Key Risk Factors (SHAP)
    if shap_explanation and 'feature_importance' in shap_explanation:
        story.append(Paragraph("Key Risk Factors", heading_style))
        
        factor_data = [['Risk Factor', 'Value', 'Contribution']]
        feature_importance = shap_explanation['feature_importance'][:8]  # Top 8
        
        for factor in feature_importance:
            factor_name = factor['feature'].replace('_', ' ').title()
            value = f"{factor['value']:.2f}"
            contribution = f"{factor['contribution']:+.3f}"
            
            factor_data.append([factor_name, value, contribution])
        
        factor_table = Table(factor_data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
        factor_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        # Color code contributions
        for i in range(1, len(factor_data)):
            contribution = float(factor_data[i][2])
            if contribution > 0:
                factor_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (2, i), (2, i), colors.red)
                ]))
            else:
                factor_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (2, i), (2, i), colors.green)
                ]))
        
        story.append(factor_table)
        story.append(Spacer(1, 20))
    
    # Care Pathway
    if care_pathway:
        story.append(PageBreak())
        story.append(Paragraph("Personalized Care Pathway", heading_style))
        
        # Risk Summary
        if 'risk_summary' in care_pathway:
            story.append(Paragraph("Risk Summary", subheading_style))
            story.append(Paragraph(care_pathway['risk_summary'], styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Primary Focus
        if 'primary_focus' in care_pathway:
            story.append(Paragraph("Primary Focus", subheading_style))
            story.append(Paragraph(care_pathway['primary_focus'], styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Phased Intervention
        if 'phased_intervention' in care_pathway:
            story.append(Paragraph("12-Week Intervention Plan", subheading_style))
            story.append(Paragraph(care_pathway['phased_intervention'], styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Lifestyle Recommendations
        if 'recommendations' in care_pathway:
            story.append(Paragraph("Lifestyle Recommendations", subheading_style))
            for rec in care_pathway['recommendations']:
                story.append(Paragraph(f"â¢ {rec}", styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Monitoring Plan
        if 'monitoring_plan' in care_pathway:
            story.append(Paragraph("Monitoring Plan", subheading_style))
            story.append(Paragraph(care_pathway['monitoring_plan'], styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Expected Outcomes
        if 'expected_outcomes' in care_pathway:
            story.append(Paragraph("Expected Outcomes", subheading_style))
            story.append(Paragraph(care_pathway['expected_outcomes'], styles['BodyText']))
            story.append(Spacer(1, 12))
    
    # Medical Disclaimer
    story.append(PageBreak())
    story.append(Paragraph("Medical Disclaimer", heading_style))
    
    disclaimer_text = """
    This HealthGuard AI report is generated by an artificial intelligence system and is intended 
    for educational and informational purposes only. It is not a substitute for professional 
    medical advice, diagnosis, or treatment.
    
    The risk assessments and recommendations provided are based on statistical models and 
    should not be used as the sole basis for medical decisions. Always seek the advice of 
    qualified healthcare providers with any questions you may have regarding a medical condition.
    
    Do not disregard professional medical advice or delay in seeking it because of something 
    you have read in this report. If you think you may have a medical emergency, call your 
    doctor or emergency services immediately.
    
    The accuracy and reliability of the AI predictions depend on the quality of input data 
    and the limitations of current medical knowledge and machine learning technology.
    """
    
    story.append(Paragraph(disclaimer_text, styles['BodyText']))
    
    # Footer information
    story.append(Spacer(1, 30))
    footer_text = f"""
    Report generated by HealthGuard AI on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}.
    
    For questions about this report, consult with your healthcare provider.
    
    Â© 2024 HealthGuard AI - Advanced Medical Risk Intelligence System
    """
    
    story.append(Paragraph(footer_text, styles['BodyText']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer

def create_download_button(buffer, filename=None):
    """Create a download button for PDF"""
    if filename is None:
        filename = f"healthguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Convert buffer to base64
    pdf_data = buffer.getvalue()
    b64 = base64.b64encode(pdf_data).decode()
    
    # Create download link
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download Health Report (PDF)</a>'
    
    return href

if __name__ == "__main__":
    # Test PDF generation
    test_patient_data = {
        'Pregnancies': 2,
        'Glucose': 145,
        'BloodPressure': 80,
        'SkinThickness': 20,
        'Insulin': 85,
        'BMI': 32.5,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 45
    }
    
    test_risk_scores = {'diabetes': 0.66, 'heart': 0.23, 'kidney': 0.15}
    test_risk_categories = {'diabetes': 'Moderate', 'heart': 'Low', 'kidney': 'Low'}
    
    buffer = generate_health_report_pdf(
        test_patient_data, 
        test_risk_scores, 
        test_risk_categories
    )
    
    # Save test PDF
    with open("test_health_report.pdf", "wb") as f:
        f.write(buffer.getvalue())
    
    print("Test PDF generated: test_health_report.pdf")
