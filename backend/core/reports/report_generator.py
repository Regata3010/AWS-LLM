# # backend/core/reports/report_generator.py

# from typing import Dict, Optional, List
# from datetime import datetime
# from pathlib import Path
# import json
# import openai
# from reportlab.lib.pagesizes import letter
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
# from reportlab.lib import colors
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
# from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
# from configurations.settings import settings
# from core.src.logger import logging

# class ComplianceReportGenerator:
#     """Generate professional compliance reports with LLM-powered summaries"""
    
#     def __init__(self):
#         # Initialize OpenAI
#         try:
#             openai.api_key = settings.OPENAI_API_KEY
#             self.client = openai
#             self.llm_available = True
#             logging.info("Report LLM initialized")
#         except:
#             logging.warning("LLM unavailable - using template summaries")
#             self.llm_available = False
        
#         # Setup styles
#         self.styles = getSampleStyleSheet()
#         self._setup_custom_styles()
    
#     def _setup_custom_styles(self):
#         """Create custom paragraph styles"""
#         # Title style
#         self.styles.add(ParagraphStyle(
#             name='CustomTitle',
#             parent=self.styles['Heading1'],
#             fontSize=24,
#             textColor=colors.HexColor('#6366F1'),
#             spaceAfter=30,
#             alignment=TA_CENTER
#         ))
        
#         # Subtitle style
#         self.styles.add(ParagraphStyle(
#             name='Subtitle',
#             parent=self.styles['Normal'],
#             fontSize=12,
#             textColor=colors.grey,
#             spaceAfter=20,
#             alignment=TA_CENTER
#         ))
        
#         # Section header
#         self.styles.add(ParagraphStyle(
#             name='SectionHeader',
#             parent=self.styles['Heading2'],
#             fontSize=16,
#             textColor=colors.HexColor('#6366F1'),
#             spaceBefore=20,
#             spaceAfter=12,
#             borderWidth=2,
#             borderColor=colors.HexColor('#6366F1'),
#             borderPadding=5
#         ))
        
#         # Body justified
#         self.styles.add(ParagraphStyle(
#             name='BodyJustified',
#             parent=self.styles['Normal'],
#             fontSize=11,
#             alignment=TA_JUSTIFY,
#             spaceAfter=12
#         ))
    
#     async def generate_report(
#         self,
#         model,
#         bias_analysis,
#         report_type: str = "compliance",
#         include_recommendations: bool = True,
#         include_technical: bool = True
#     ) -> Dict:
#         """Generate complete report data and PDF"""
        
#         logging.info(f"Generating {report_type} report for {model.model_id}")
        
#         # Parse metrics
#         if isinstance(bias_analysis.fairness_metrics, str):
#             fairness_metrics = json.loads(bias_analysis.fairness_metrics)
#         else:
#             fairness_metrics = bias_analysis.fairness_metrics
        
#         # Generate LLM content
#         executive_summary = await self._generate_executive_summary(
#             model, bias_analysis, fairness_metrics
#         )
        
#         recommendations = await self._generate_recommendations(
#             model, bias_analysis, fairness_metrics
#         ) if include_recommendations else []
        
#         # Prepare report data
#         report_data = {
#             "report_id": f"RPT-{model.model_id}-{datetime.now().strftime('%Y%m%d')}",
#             "generated_at": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
#             "model": model,
#             "bias_analysis": bias_analysis,
#             "fairness_metrics": fairness_metrics,
#             "executive_summary": executive_summary,
#             "recommendations": recommendations,
#             "include_technical": include_technical
#         }
        
#         return {
#             "report_data": report_data,
#             "executive_summary": executive_summary
#         }
    
#     def create_pdf(self, report_data: Dict, output_path: Path):
#         """Create PDF using ReportLab"""
        
#         doc = SimpleDocTemplate(
#             str(output_path),
#             pagesize=letter,
#             rightMargin=72,
#             leftMargin=72,
#             topMargin=72,
#             bottomMargin=72
#         )
        
#         story = []
        
#         # Cover Page
#         story.extend(self._build_cover_page(report_data))
        
#         # Executive Summary
#         story.extend(self._build_executive_summary(report_data))
        
#         # Compliance Status
#         story.extend(self._build_compliance_section(report_data))
        
#         # Fairness Metrics
#         story.extend(self._build_metrics_section(report_data))
        
#         # Recommendations
#         if report_data["recommendations"]:
#             story.extend(self._build_recommendations_section(report_data))
        
#         # Page break before technical details
#         story.append(PageBreak())
        
#         # Regulatory Compliance
#         story.extend(self._build_regulatory_section(report_data))
        
#         # Technical Details
#         if report_data["include_technical"]:
#             story.extend(self._build_technical_section(report_data))
        
#         # Footer
#         story.extend(self._build_footer())
        
#         # Build PDF
#         doc.build(story)
#         logging.info(f"PDF created: {output_path}")
    
#     def _build_cover_page(self, data: Dict) -> List:
#         """Build cover page elements"""
#         elements = []
        
#         # Title
#         elements.append(Spacer(1, 1*inch))
#         elements.append(Paragraph(
#             "BiasGuard",
#             self.styles['CustomTitle']
#         ))
        
#         # Subtitle
#         elements.append(Paragraph(
#             "AI Fairness Compliance Report",
#             self.styles['Subtitle']
#         ))
        
#         elements.append(Spacer(1, 0.5*inch))
        
#         # Metadata table
#         metadata = [
#             ['Report ID:', data['report_id']],
#             ['Generated:', data['generated_at']],
#             ['Model ID:', data['model'].model_id],
#             ['Model Type:', data['model'].model_type],
#             ['Compliance Status:', data['bias_analysis'].compliance_status]
#         ]
        
#         table = Table(metadata, colWidths=[2*inch, 4*inch])
#         table.setStyle(TableStyle([
#             ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
#             ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
#             ('FONTSIZE', (0, 0), (-1, -1), 11),
#             ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6b7280')),
#             ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
#             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
#             ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
#         ]))
        
#         elements.append(table)
#         elements.append(PageBreak())
        
#         return elements
    
#     def _build_executive_summary(self, data: Dict) -> List:
#         """Build executive summary section"""
#         elements = []
        
#         elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
#         elements.append(Spacer(1, 0.2*inch))
        
#         # Summary paragraphs
#         summary_paragraphs = data['executive_summary'].split('\n\n')
#         for para in summary_paragraphs:
#             if para.strip():
#                 elements.append(Paragraph(para.strip(), self.styles['BodyJustified']))
#                 elements.append(Spacer(1, 0.1*inch))
        
#         elements.append(Spacer(1, 0.3*inch))
        
#         return elements
    
#     def _build_compliance_section(self, data: Dict) -> List:
#         """Build compliance status section"""
#         elements = []
        
#         elements.append(Paragraph("Compliance Status", self.styles['SectionHeader']))
#         elements.append(Spacer(1, 0.2*inch))
        
#         status = data['bias_analysis'].compliance_status
#         is_compliant = "COMPLIANT" in status
        
#         # Status box
#         status_data = [[
#             Paragraph(f"<b>{status}</b>", self.styles['Normal']),
#             Paragraph(f"Analyzed: {data['bias_analysis'].analyzed_at}", self.styles['Normal'])
#         ]]
        
#         table = Table(status_data, colWidths=[4*inch, 2*inch])
#         table.setStyle(TableStyle([
#             ('BACKGROUND', (0, 0), (-1, -1), 
#              colors.HexColor('#d1fae5') if is_compliant else colors.HexColor('#fee2e2')),
#             ('TEXTCOLOR', (0, 0), (-1, -1),
#              colors.HexColor('#065f46') if is_compliant else colors.HexColor('#991b1b')),
#             ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
#             ('FONTSIZE', (0, 0), (-1, -1), 12),
#             ('PADDING', (0, 0), (-1, -1), 15),
#             ('BORDER', (0, 0), (-1, -1), 2,
#              colors.HexColor('#10b981') if is_compliant else colors.HexColor('#ef4444'))
#         ]))
        
#         elements.append(table)
#         elements.append(Spacer(1, 0.3*inch))
        
#         return elements
    
#     def _build_metrics_section(self, data: Dict) -> List:
#         """Build fairness metrics table"""
#         elements = []
        
#         elements.append(Paragraph("Fairness Metrics", self.styles['SectionHeader']))
#         elements.append(Spacer(1, 0.2*inch))
        
#         # Metrics table data
#         table_data = [['Metric', 'Sensitive Attr', 'Value', 'Threshold', 'Status']]
        
#         for sensitive_attr, metrics in data['fairness_metrics'].items():
#             for metric_name, metric_data in metrics.items():
#                 if isinstance(metric_data, dict):
#                     value = metric_data.get('ratio') or metric_data.get('difference', 0)
#                     threshold = metric_data.get('threshold', 'N/A')
#                     status = metric_data.get('severity', 'UNKNOWN')
                    
#                     table_data.append([
#                         metric_name.replace('_', ' ').title(),
#                         sensitive_attr,
#                         f"{value:.4f}" if isinstance(value, (int, float)) else str(value),
#                         str(threshold),
#                         status
#                     ])
        
#         table = Table(table_data, colWidths=[1.8*inch, 1.3*inch, 1*inch, 1.2*inch, 1*inch])
#         table.setStyle(TableStyle([
#             ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366F1')),
#             ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
#             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#             ('FONTSIZE', (0, 0), (-1, 0), 11),
#             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#             ('BACKGROUND', (0, 1), (-1, -1), colors.white),
#             ('GRID', (0, 0), (-1, -1), 1, colors.grey),
#             ('FONTSIZE', (0, 1), (-1, -1), 10),
#             ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
#         ]))
        
#         elements.append(table)
#         elements.append(Spacer(1, 0.3*inch))
        
#         return elements
    
#     def _build_recommendations_section(self, data: Dict) -> List:
#         """Build recommendations section"""
#         elements = []
        
#         elements.append(Paragraph("Recommendations", self.styles['SectionHeader']))
#         elements.append(Spacer(1, 0.2*inch))
        
#         for i, rec in enumerate(data['recommendations'], 1):
#             # Color code by priority
#             if 'CRITICAL' in rec:
#                 color = '#ef4444'
#             elif 'HIGH' in rec:
#                 color = '#f59e0b'
#             else:
#                 color = '#6b7280'
            
#             elements.append(Paragraph(
#                 f'<font color="{color}"><b>{i}.</b></font> {rec}',
#                 self.styles['Normal']
#             ))
#             elements.append(Spacer(1, 0.15*inch))
        
#         elements.append(Spacer(1, 0.3*inch))
        
#         return elements
    
#     def _build_regulatory_section(self, data: Dict) -> List:
#         """Build regulatory compliance checklist"""
#         elements = []
        
#         elements.append(Paragraph("Regulatory Compliance Checklist", self.styles['SectionHeader']))
#         elements.append(Spacer(1, 0.2*inch))
        
#         checks = self._get_regulatory_checks(data['bias_analysis'], data['fairness_metrics'])
        
#         # Checklist table
#         table_data = [['Regulation', 'Requirement', 'Status', 'Notes']]
        
#         for check in checks:
#             table_data.append([
#                 check['regulation'],
#                 check['requirement'],
#                 check['status'],
#                 check['notes']
#             ])
        
#         table = Table(table_data, colWidths=[1.5*inch, 2*inch, 0.8*inch, 2*inch])
#         table.setStyle(TableStyle([
#             ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366F1')),
#             ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
#             ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
#             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#             ('FONTSIZE', (0, 0), (-1, 0), 10),
#             ('FONTSIZE', (0, 1), (-1, -1), 9),
#             ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
#             ('GRID', (0, 0), (-1, -1), 1, colors.grey),
#             ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
#             ('VALIGN', (0, 0), (-1, -1), 'TOP')
#         ]))
        
#         elements.append(table)
#         elements.append(Spacer(1, 0.3*inch))
        
#         return elements
    
#     def _build_technical_section(self, data: Dict) -> List:
#         """Build technical details section"""
#         elements = []
        
#         elements.append(Paragraph("Technical Details", self.styles['SectionHeader']))
#         elements.append(Spacer(1, 0.2*inch))
        
#         model = data['model']
        
#         tech_data = [
#             ['Model Architecture:', model.model_type],
#             ['Task Type:', model.task_type],
#             ['Training Samples:', f"{model.training_samples:,}"],
#             ['Test Samples:', f"{model.test_samples:,}"],
#             ['Accuracy:', f"{model.accuracy * 100:.2f}%"],
#             ['Target Variable:', model.target_column],
#             ['Protected Attributes:', ', '.join(model.sensitive_columns)],
#             ['MLflow Run ID:', model.mlflow_run_id if model.mlflow_run_id else 'N/A']
#         ]
        
#         table = Table(tech_data, colWidths=[2*inch, 4*inch])
#         table.setStyle(TableStyle([
#             ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
#             ('FONTSIZE', (0, 0), (-1, -1), 10),
#             ('PADDING', (0, 0), (-1, -1), 8),
#             ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f9fafb')),
#             ('GRID', (0, 0), (-1, -1), 1, colors.grey)
#         ]))
        
#         elements.append(table)
#         elements.append(Spacer(1, 0.3*inch))
        
#         return elements
    
#     def _build_footer(self) -> List:
#         """Build report footer"""
#         elements = []
        
#         elements.append(Spacer(1, 0.5*inch))
        
#         footer_text = """
#         <para align="center" fontSize="9" textColor="#6b7280">
#         This report was automatically generated by BiasGuard AI Fairness Platform<br/>
#         For questions or support, contact: compliance@biasguard.ai<br/><br/>
#         © 2025 BiasGuard. All rights reserved.
#         </para>
#         """
        
#         elements.append(Paragraph(footer_text, self.styles['Normal']))
        
#         return elements
    
#     async def _generate_executive_summary(self, model, bias_analysis, metrics) -> str:
#         """Generate executive summary with LLM"""
        
#         if not self.llm_available:
#             return self._template_executive_summary(model, bias_analysis)
        
#         context = f"""Generate an executive summary for this AI fairness compliance report:

# MODEL: {model.model_type}, Accuracy: {model.accuracy * 100:.2f}%
# COMPLIANCE STATUS: {bias_analysis.compliance_status}
# SENSITIVE ATTRIBUTES: {', '.join(model.sensitive_columns)}
# SAMPLES: {model.training_samples + model.test_samples:,}

# FAIRNESS METRICS SUMMARY:
# {json.dumps(metrics, indent=2)[:500]}

# Write a professional 3-paragraph executive summary for a compliance officer:
# 1. Overall compliance status and key finding
# 2. Specific metric highlights (2-3 items)
# 3. Clear recommendation

# Be concise, regulatory-focused, and data-driven. Max 250 words."""

#         try:
#             response = self.client.ChatCompletion.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a compliance officer writing AI fairness reports for CFPB audits."},
#                     {"role": "user", "content": context}
#                 ],
#                 temperature=0.3,
#                 max_completion_tokens=400
#             )
            
#             return response.choices[0].message.content.strip()
            
#         except Exception as e:
#             logging.error(f"LLM summary failed: {e}")
#             return self._template_executive_summary(model, bias_analysis)
    
#     async def _generate_recommendations(self, model, bias_analysis, metrics) -> List[str]:
#         """Generate recommendations with LLM"""
        
#         if not self.llm_available:
#             return bias_analysis.recommendations if bias_analysis.recommendations else []
        
#         context = f"""Generate 4-5 prioritized recommendations for this AI model:

# STATUS: {bias_analysis.compliance_status}
# CURRENT RECS: {json.dumps(bias_analysis.recommendations)}

# Format: "[PRIORITY]: Action item (brief reasoning)"
# Priority levels: CRITICAL, HIGH, MEDIUM

# Be specific and actionable."""

#         try:
#             response = self.client.ChatCompletion.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are an AI fairness consultant."},
#                     {"role": "user", "content": context}
#                 ],
#                 temperature=0.3,
#                 max_completion_tokens=300
#             )
            
#             text = response.choices[0].message.content.strip()
#             return [r.strip() for r in text.split('\n') if r.strip()]
            
#         except:
#             return bias_analysis.recommendations if bias_analysis.recommendations else []
    
#     def _template_executive_summary(self, model, bias_analysis) -> str:
#         """Fallback template summary"""
#         status = bias_analysis.compliance_status
#         is_compliant = "COMPLIANT" in status
        
#         if is_compliant:
#             return f"""This report presents bias analysis for the {model.model_type} model (ID: {model.model_id}). The model demonstrates COMPLIANT fairness levels across all tested metrics, meeting CFPB and ECOA requirements for {', '.join(model.sensitive_columns)}.

# With {model.accuracy * 100:.2f}% accuracy, the model maintains strong performance while adhering to fairness constraints. All metrics fall within acceptable thresholds.

# The model is approved for production deployment subject to continued monitoring per CFPB AI/ML guidelines."""
#         else:
#             return f"""This report presents bias analysis for the {model.model_type} model (ID: {model.model_id}). The analysis identified fairness concerns requiring attention.

# Status: {status}. The model shows bias indicators across {', '.join(model.sensitive_columns)} while maintaining {model.accuracy * 100:.2f}% accuracy.

# RECOMMENDATION: Apply bias mitigation before deployment."""
    
#     def _get_regulatory_checks(self, bias_analysis, metrics) -> List[Dict]:
#         """Generate regulatory checklist"""
#         return [
#             {
#                 "regulation": "CFPB AI/ML Guidelines",
#                 "requirement": "Regular bias audits",
#                 "status": "PASS",
#                 "notes": "Analysis performed with 7 fairness metrics"
#             },
#             {
#                 "regulation": "ECOA (80% Rule)",
#                 "requirement": "No disparate impact",
#                 "status": self._check_disparate_impact(metrics),
#                 "notes": "Disparate impact ratio evaluated"
#             },
#             {
#                 "regulation": "Title VII",
#                 "requirement": "No discrimination",
#                 "status": self._check_statistical_parity(metrics),
#                 "notes": "Statistical parity measured"
#             },
#             {
#                 "regulation": "Explainability",
#                 "requirement": "Adverse action notices",
#                 "status": "PASS",
#                 "notes": "MLflow tracking enabled"
#             }
#         ]
    
#     def _check_disparate_impact(self, metrics) -> str:
#         for attr_metrics in metrics.values():
#             if 'disparate_impact' in attr_metrics:
#                 di = attr_metrics['disparate_impact']
#                 ratio = di.get('ratio', 1.0)
                
#                 if isinstance(ratio, str):
#                     return "REVIEW"
                
#                 if ratio < 0.8 or ratio > 1.25:
#                     return "FAIL"
        
#         return "PASS"
    
#     def _check_statistical_parity(self, metrics) -> str:
#         for attr_metrics in metrics.values():
#             if 'statistical_parity' in attr_metrics:
#                 sp = attr_metrics['statistical_parity']
#                 diff = abs(sp.get('statistical_parity_diff', 0))
                
#                 if diff > 0.1:
#                     return "FAIL"
        
#         return "PASS"


#-----------------------------------------------------------------------------------------------------------------------------

# backend/core/reports/report_generator.py
# FIXED VERSION - Properly extracts nested fairness metrics

from typing import Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import os
from core.src.logger import logging

class ComplianceReportGenerator:
    """
    Generate comprehensive compliance reports for CFPB and other regulators
    FIXED: Properly handles nested fairness metrics structure
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for professional reports"""
        
        # Helper function to add style if it doesn't exist
        def add_style_if_not_exists(name, style_params):
            if name not in self.styles:
                self.styles.add(ParagraphStyle(name=name, **style_params))
        
        # Title style
        add_style_if_not_exists('CustomTitle', {
            'parent': self.styles['Heading1'],
            'fontSize': 28,
            'textColor': colors.HexColor('#1a237e'),
            'spaceAfter': 20,
            'alignment': TA_CENTER,
            'fontName': 'Helvetica-Bold'
        })
        
        # Report subtitle
        add_style_if_not_exists('Subtitle', {
            'parent': self.styles['Normal'],
            'fontSize': 14,
            'textColor': colors.HexColor('#5e35b1'),
            'spaceAfter': 30,
            'alignment': TA_CENTER,
            'fontName': 'Helvetica'
        })
        
        # Section headers
        add_style_if_not_exists('SectionHeader', {
            'parent': self.styles['Heading2'],
            'fontSize': 16,
            'textColor': colors.HexColor('#3f51b5'),
            'spaceAfter': 12,
            'spaceBefore': 20,
            'fontName': 'Helvetica-Bold'
        })
        
        # Warning text
        add_style_if_not_exists('Warning', {
            'parent': self.styles['Normal'],
            'fontSize': 11,
            'textColor': colors.HexColor('#ff5722'),
            'leftIndent': 20,
            'fontName': 'Helvetica-Bold'
        })
        
        # Success text
        add_style_if_not_exists('Success', {
            'parent': self.styles['Normal'],
            'fontSize': 11,
            'textColor': colors.HexColor('#4caf50'),
            'leftIndent': 20,
            'fontName': 'Helvetica-Bold'
        })
        
        # Info text
        add_style_if_not_exists('Info', {
            'parent': self.styles['Normal'],
            'fontSize': 11,
            'textColor': colors.HexColor('#2196f3'),
            'leftIndent': 20
        })
        
        # Body text
        add_style_if_not_exists('BodyText', {
            'parent': self.styles['Normal'],
            'fontSize': 11,
            'leading': 16,
            'alignment': TA_JUSTIFY,
            'spaceAfter': 10
        })
    
    async def generate_report(
        self,
        model,
        bias_analysis,
        report_type: str = "compliance",
        include_recommendations: bool = True,
        include_technical: bool = True
    ) -> Dict:
        """
        Generate comprehensive compliance report
        
        Args:
            model: Model database object
            bias_analysis: BiasAnalysis database object
            report_type: Type of report (compliance, technical, executive)
            include_recommendations: Include recommendations section
            include_technical: Include technical details
            
        Returns:
            Dict with report_data and executive_summary
        """
        
        logging.info(f"Generating {report_type} report for model {model.model_id}")
        
        # Parse fairness metrics
        if isinstance(bias_analysis.fairness_metrics, str):
            fairness_metrics = json.loads(bias_analysis.fairness_metrics)
        else:
            fairness_metrics = bias_analysis.fairness_metrics
        
        # Parse AIF360 metrics
        aif360_metrics = None
        if bias_analysis.aif360_metrics:
            if isinstance(bias_analysis.aif360_metrics, str):
                aif360_metrics = json.loads(bias_analysis.aif360_metrics)
            else:
                aif360_metrics = bias_analysis.aif360_metrics
        
        # Build report data
        report_data = {
            "metadata": self._build_metadata(model, bias_analysis),
            "executive_summary": self._generate_executive_summary(model, fairness_metrics, bias_analysis),
            "compliance_status": self._get_compliance_status(model, fairness_metrics),
            "fairness_metrics": self._prepare_detailed_metrics(fairness_metrics),
            "aif360_metrics": aif360_metrics,
            "recommendations": bias_analysis.recommendations if include_recommendations else [],
            "regulatory_compliance": self._generate_compliance_section(model, fairness_metrics),
            "technical_details": self._build_technical_details(model) if include_technical else None
        }
        
        return {
            "report_data": report_data,
            "executive_summary": report_data["executive_summary"]
        }
    
    def create_pdf(self, report_data: Dict, output_path: Path):
        """
        Create PDF from report data using ReportLab
        
        Args:
            report_data: Report data dictionary
            output_path: Path to save PDF
        """
        
        logging.info(f"Creating PDF report: {output_path}")
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=36
        )
        
        # Build content
        story = []
        
        # ============================================
        # TITLE PAGE
        # ============================================
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("BiasGuard", self.styles['CustomTitle']))
        story.append(Paragraph("AI Fairness Compliance Report", self.styles['Subtitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Report metadata table
        metadata = report_data["metadata"]
        metadata_data = [
            ["Report ID:", metadata["report_id"]],
            ["Generated:", metadata["generated_at"]],
            ["Model ID:", metadata["model_id"]],
            ["Model Type:", metadata["model_type"]],
            ["Compliance Status:", metadata["compliance_status"]]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8eaf6')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#283593')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c5cae9'))
        ]))
        
        story.append(metadata_table)
        story.append(PageBreak())
        
        # ============================================
        # EXECUTIVE SUMMARY
        # ============================================
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        story.append(Spacer(1, 12))
        
        summary_paragraphs = report_data["executive_summary"].split('\n\n')
        for para in summary_paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), self.styles['BodyText']))
        
        story.append(Spacer(1, 20))
        
        # Compliance status banner
        compliance = report_data["compliance_status"]
        status_color = self._get_status_color(compliance["status"])
        
        status_table = Table(
            [[Paragraph(f"<b>Compliance Status:</b> {compliance['status']}", self.styles['Normal'])]],
            colWidths=[6*inch]
        )
        status_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), status_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        story.append(status_table)
        story.append(Spacer(1, 12))
        
        analyzed_date = metadata.get("analyzed_at", "N/A")
        story.append(Paragraph(f"<i>Analyzed: {analyzed_date}</i>", self.styles['Normal']))
        
        story.append(PageBreak())
        
        # ============================================
        # FAIRNESS METRICS
        # ============================================
        story.append(Paragraph("Fairness Metrics", self.styles['SectionHeader']))
        story.append(Spacer(1, 12))
        
        metrics_table = self._create_metrics_table(report_data["fairness_metrics"])
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Create visualization
        chart_path = self._create_metrics_chart(report_data["fairness_metrics"])
        if chart_path and os.path.exists(chart_path):
            try:
                story.append(Image(chart_path, width=6*inch, height=3.5*inch))
                story.append(Spacer(1, 12))
            except Exception as e:
                logging.error(f"Failed to add chart to PDF: {e}")
        
        story.append(PageBreak())
        
        # ============================================
        # RECOMMENDATIONS
        # ============================================
        if report_data["recommendations"]:
            story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
            story.append(Spacer(1, 12))
            
            for i, rec in enumerate(report_data["recommendations"], 1):
                priority = "CRITICAL" if "CRITICAL" in rec else "HIGH" if "HIGH" in rec else "MEDIUM"
                priority_style = self.styles['Warning'] if priority == "CRITICAL" else self.styles['Info']
                
                story.append(Paragraph(f"{i}. {rec}", priority_style))
                story.append(Spacer(1, 8))
            
            story.append(PageBreak())
        
        # ============================================
        # REGULATORY COMPLIANCE
        # ============================================
        story.append(Paragraph("Regulatory Compliance Checklist", self.styles['SectionHeader']))
        story.append(Spacer(1, 12))
        
        regulatory_table = self._create_regulatory_table(report_data["regulatory_compliance"])
        story.append(regulatory_table)
        story.append(Spacer(1, 20))
        
        # ============================================
        # TECHNICAL DETAILS
        # ============================================
        if report_data.get("technical_details"):
            story.append(PageBreak())
            story.append(Paragraph("Technical Details", self.styles['SectionHeader']))
            story.append(Spacer(1, 12))
            
            tech_table = self._create_technical_table(report_data["technical_details"])
            story.append(tech_table)
        
        # ============================================
        # FOOTER
        # ============================================
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph(
            "This report was automatically generated by BiasGuard AI Fairness Platform",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            "For questions or support, contact: compliance@biasguard.ai",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            "© 2025 BiasGuard. All rights reserved.",
            self.styles['Normal']
        ))
        
        # Build PDF
        doc.build(story)
        logging.info(f"✅ PDF report created: {output_path}")
    
    def _build_metadata(self, model, bias_analysis) -> Dict:
        """Build report metadata"""
        return {
            "report_id": f"RPT-{model.model_id}-{datetime.now().strftime('%Y%m%d')}",
            "generated_at": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            "model_id": model.model_id,
            "model_type": model.model_type,
            "compliance_status": bias_analysis.compliance_status,
            "analyzed_at": bias_analysis.analyzed_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(bias_analysis.analyzed_at, 'strftime') else str(bias_analysis.analyzed_at)
        }
    
    def _generate_executive_summary(self, model, fairness_metrics: Dict, bias_analysis) -> str:
        """Generate executive summary with actual metric values"""
        
        # Get first sensitive attribute's metrics
        if not fairness_metrics:
            return "No bias analysis data available for this model."
        
        first_attr = list(fairness_metrics.keys())[0]
        attr_metrics = fairness_metrics[first_attr]
        
        # Extract key metrics
        di_data = attr_metrics.get('disparate_impact', {})
        di_ratio = di_data.get('ratio', 1.0)
        if di_ratio in ['inf', float('inf')]:
            di_ratio = 999.0
        else:
            di_ratio = float(di_ratio)
        
        sp_data = attr_metrics.get('statistical_parity', {})
        sp_diff = abs(sp_data.get('statistical_parity_diff', 0))
        
        eo_data = attr_metrics.get('equal_opportunity', {})
        eo_diff = abs(eo_data.get('difference', 0))
        
        # Determine risk level
        if di_ratio < 0.8 or di_ratio > 1.25:
            risk_level = "high risk"
            action = "immediate attention required"
        elif di_ratio < 0.85 or di_ratio > 1.20:
            risk_level = "moderate risk"
            action = "monitoring and mitigation recommended"
        else:
            risk_level = "low risk"
            action = "continue monitoring"
        
        # Build summary
        summary = f"""
The AI fairness compliance report for the {model.model_type} model, which achieved an accuracy of {model.accuracy * 100:.2f}%, indicates a compliance status of "{bias_analysis.compliance_status}". The analysis, based on a sample size of {model.training_samples + model.test_samples:,}, highlights potential disparities in prediction outcomes across sensitive attributes, including {', '.join(model.sensitive_columns)}.

Key metrics reveal that while the statistical parity difference for {first_attr} is {sp_diff:.4f}, {'indicating no significant bias' if sp_diff < 0.1 else 'indicating potential bias'}, the equal opportunity metric shows {'a slight' if eo_diff < 0.1 else 'a significant'} disparity in true positive rates between groups. The disparate impact ratio of {di_ratio:.4f} {'meets' if 0.8 <= di_ratio <= 1.25 else 'does not meet'} the CFPB 80% rule threshold.

Current assessment indicates {risk_level} of discriminatory outcomes. Action recommended: {action}.

To address these concerns, it is recommended that the model undergoes {'further refinement' if di_ratio < 0.8 else 'continued monitoring'} to {'reduce' if di_ratio < 0.8 else 'maintain'} bias. This may include implementing bias mitigation strategies, such as re-weighting training samples or incorporating fairness constraints during model training. Additionally, ongoing monitoring of fairness metrics is essential to ensure compliance with regulatory expectations and to foster equitable outcomes across all demographic groups.
        """.strip()
        
        return summary
    
    def _get_compliance_status(self, model, fairness_metrics: Dict) -> Dict:
        """Determine detailed compliance status"""
        
        violations = []
        warnings = []
        
        for attr_name, metrics in fairness_metrics.items():
            # Check disparate impact
            di_data = metrics.get('disparate_impact', {})
            di_ratio = di_data.get('ratio', 1.0)
            
            if di_ratio not in ['inf', float('inf')]:
                di_ratio = float(di_ratio)
                if di_ratio < 0.8 or di_ratio > 1.25:
                    violations.append(f"{attr_name}: Disparate Impact = {di_ratio:.3f}")
                elif di_ratio < 0.85 or di_ratio > 1.20:
                    warnings.append(f"{attr_name}: Disparate Impact = {di_ratio:.3f}")
            
            # Check statistical parity
            sp_data = metrics.get('statistical_parity', {})
            sp_diff = abs(sp_data.get('statistical_parity_diff', 0))
            
            if sp_diff > 0.2:
                violations.append(f"{attr_name}: Statistical Parity = {sp_diff:.3f}")
            elif sp_diff > 0.1:
                warnings.append(f"{attr_name}: Statistical Parity = {sp_diff:.3f}")
        
        # Determine overall status
        if violations:
            status = "NON-COMPLIANT"
        elif warnings:
            status = "WARNING"
        else:
            status = "COMPLIANT"
        
        # Check for intersectional bias override
        if model.bias_status == "warning" and "Intersectional" in model.bias_status:
            status = "WARNING - Intersectional bias detected"
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings
        }
    
    def _prepare_detailed_metrics(self, fairness_metrics: Dict) -> List[Dict]:
        """
        Prepare detailed metrics for table display
        FIXED: Properly extracts nested metric values
        """
        
        metrics_list = []
        
        for sensitive_attr, metrics_dict in fairness_metrics.items():
            
            # Statistical Parity
            if 'statistical_parity' in metrics_dict:
                sp = metrics_dict['statistical_parity']
                metrics_list.append({
                    "metric": "Statistical Parity",
                    "sensitive_attr": sensitive_attr,
                    "value": abs(sp.get('statistical_parity_diff', 0)),
                    "threshold": sp.get('threshold', 0.1),
                    "status": sp.get('severity', 'UNKNOWN'),
                    "group_0_rate": sp.get('group_0_rate', 0),
                    "group_1_rate": sp.get('group_1_rate', 0)
                })
            
            # Equal Opportunity
            if 'equal_opportunity' in metrics_dict:
                eo = metrics_dict['equal_opportunity']
                metrics_list.append({
                    "metric": "Equal Opportunity",
                    "sensitive_attr": sensitive_attr,
                    "value": abs(eo.get('difference', 0)),
                    "threshold": eo.get('threshold', 0.1),
                    "status": eo.get('severity', 'UNKNOWN'),
                    "group_0_rate": eo.get('group_0_tpr', 0),
                    "group_1_rate": eo.get('group_1_tpr', 0)
                })
            
            # Disparate Impact
            if 'disparate_impact' in metrics_dict:
                di = metrics_dict['disparate_impact']
                ratio = di.get('ratio', 1.0)
                metrics_list.append({
                    "metric": "Disparate Impact",
                    "sensitive_attr": sensitive_attr,
                    "value": ratio if ratio not in ['inf', float('inf')] else 'inf',
                    "threshold": str(di.get('threshold', [0.8, 1.2])),
                    "status": di.get('severity', 'UNKNOWN'),
                    "group_0_rate": di.get('group_0_rate', 0),
                    "group_1_rate": di.get('group_1_rate', 0)
                })
            
            # Average Odds
            if 'average_odds' in metrics_dict:
                ao = metrics_dict['average_odds']
                metrics_list.append({
                    "metric": "Average Odds",
                    "sensitive_attr": sensitive_attr,
                    "value": abs(ao.get('average_difference', 0)),
                    "threshold": ao.get('threshold', 0.1),
                    "status": ao.get('severity', 'UNKNOWN')
                })
            
            # FPR Parity
            if 'fpr_parity' in metrics_dict:
                fpr = metrics_dict['fpr_parity']
                metrics_list.append({
                    "metric": "Fpr Parity",
                    "sensitive_attr": sensitive_attr,
                    "value": abs(fpr.get('fpr_difference', 0)),
                    "threshold": fpr.get('threshold', 0.1),
                    "status": fpr.get('severity', 'UNKNOWN')
                })
            
            # Predictive Parity
            if 'predictive_parity' in metrics_dict:
                pp = metrics_dict['predictive_parity']
                metrics_list.append({
                    "metric": "Predictive Parity",
                    "sensitive_attr": sensitive_attr,
                    "value": abs(pp.get('ppv_difference', 0)),
                    "threshold": pp.get('threshold', 0.1),
                    "status": pp.get('severity', 'UNKNOWN')
                })
            
            # Treatment Equality
            if 'treatment_equality' in metrics_dict:
                te = metrics_dict['treatment_equality']
                ratio_diff = te.get('ratio_difference', 0)
                metrics_list.append({
                    "metric": "Treatment Equality",
                    "sensitive_attr": sensitive_attr,
                    "value": ratio_diff if ratio_diff not in ['inf', float('inf')] else 'inf',
                    "threshold": te.get('threshold', 1.0),
                    "status": te.get('severity', 'UNKNOWN')
                })
        
        return metrics_list
    
    def _create_metrics_table(self, metrics_list: List[Dict]) -> Table:
        """Create formatted metrics table"""
        
        # Table header
        data = [["Metric", "Sensitive Attr", "Value", "Threshold", "Status"]]
        
        # Add metric rows
        for metric in metrics_list:
            value = metric["value"]
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            
            data.append([
                metric["metric"],
                metric["sensitive_attr"],
                value_str,
                str(metric["threshold"]),
                metric["status"]
            ])
        
        # Create table
        table = Table(data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch, 1*inch])
        
        # Style table
        table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Body
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
        ]))
        
        return table
    
    def _create_metrics_chart(self, metrics_list: List[Dict]) -> Optional[str]:
        """
        Create visualization of fairness metrics
        FIXED: Uses actual metric values
        """
        try:
            # Extract metrics for visualization
            chart_data = {}
            
            for metric in metrics_list:
                if metric["metric"] in ["Statistical Parity", "Equal Opportunity", "Average Odds"]:
                    value = metric["value"]
                    if isinstance(value, (int, float)):
                        label = f"{metric['metric'][:2]} ({metric['sensitive_attr'][:3]})"
                        chart_data[label] = value
            
            if not chart_data:
                logging.warning("No data available for chart")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Determine colors based on thresholds
            colors_list = []
            for value in chart_data.values():
                if abs(value) < 0.1:
                    colors_list.append('#4caf50')  # Green - good
                elif abs(value) < 0.2:
                    colors_list.append('#ff9800')  # Orange - warning
                else:
                    colors_list.append('#f44336')  # Red - critical
            
            # Create bar chart
            bars = ax.bar(chart_data.keys(), chart_data.values(), color=colors_list, alpha=0.8, edgecolor='black')
            
            # Add threshold lines
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Warning Threshold')
            ax.axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Critical Threshold')
            ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
            
            # Labels and title
            ax.set_ylabel('Difference from Parity', fontsize=12, fontweight='bold')
            ax.set_xlabel('Fairness Metrics', fontsize=12, fontweight='bold')
            ax.set_title('Fairness Metrics Overview', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save to file
            chart_dir = Path("reports/charts")
            chart_dir.mkdir(parents=True, exist_ok=True)
            chart_path = chart_dir / f"metrics_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logging.info(f"Chart created: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logging.error(f"Error creating metrics chart: {e}", exc_info=True)
            return None
    
    def _generate_compliance_section(self, model, fairness_metrics: Dict) -> Dict:
        """Generate regulatory compliance checklist"""
        
        # Get first attribute metrics for compliance check
        first_attr = list(fairness_metrics.keys())[0] if fairness_metrics else None
        
        if not first_attr:
            return {
                "cfpb": {"status": "UNKNOWN", "notes": "No analysis available"},
                "ecoa": {"status": "UNKNOWN", "notes": "No analysis available"},
                "title_vii": {"status": "UNKNOWN", "notes": "No analysis available"},
                "explainability": {"status": "UNKNOWN", "notes": "No analysis available"}
            }
        
        attr_metrics = fairness_metrics[first_attr]
        
        # Check CFPB (Disparate Impact)
        di_data = attr_metrics.get('disparate_impact', {})
        di_ratio = di_data.get('ratio', 1.0)
        if di_ratio in ['inf', float('inf')]:
            cfpb_status = "FAIL"
            cfpb_notes = "Infinite disparate impact - one group has no favorable outcomes"
        elif 0.8 <= di_ratio <= 1.25:
            cfpb_status = "PASS"
            cfpb_notes = "Disparate impact ratio within acceptable range"
        else:
            cfpb_status = "FAIL"
            cfpb_notes = f"Disparate impact ratio {di_ratio:.3f} outside 0.8-1.25 range"
        
        # Check ECOA (80% rule - same as disparate impact)
        ecoa_status = "PASS" if cfpb_status == "PASS" else "FAIL"
        ecoa_notes = "Disparate impact ratio evaluated" if cfpb_status == "PASS" else "80% rule violation"
        
        # Check Title VII (Statistical Parity)
        sp_data = attr_metrics.get('statistical_parity', {})
        sp_diff = abs(sp_data.get('statistical_parity_diff', 0))
        title_vii_status = "PASS" if sp_diff < 0.1 else "FAIL"
        title_vii_notes = "Statistical parity measured" if sp_diff < 0.1 else f"Statistical parity difference {sp_diff:.3f} exceeds 0.1"
        
        # Check Explainability (MLflow tracking)
        explainability_status = "PASS" if model.mlflow_run_id else "FAIL"
        explainability_notes = "MLflow tracking enabled" if model.mlflow_run_id else "No MLflow tracking"
        
        return {
            "cfpb": {"requirement": "Regular bias audits", "status": cfpb_status, "notes": cfpb_notes},
            "ecoa": {"requirement": "No disparate impact", "status": ecoa_status, "notes": ecoa_notes},
            "title_vii": {"requirement": "No discrimination", "status": title_vii_status, "notes": title_vii_notes},
            "explainability": {"requirement": "Adverse action notices", "status": explainability_status, "notes": explainability_notes}
        }
    
    def _create_regulatory_table(self, compliance_data: Dict) -> Table:
        """Create regulatory compliance checklist table"""
        
        data = [["Regulation", "Requirement", "Status", "Notes"]]
        
        regulations = {
            "cfpb": "CFPB AI/ML Guidelines",
            "ecoa": "ECOA (80% Rule)",
            "title_vii": "Title VII",
            "explainability": "Explainability"
        }
        
        for key, name in regulations.items():
            if key in compliance_data:
                comp = compliance_data[key]
                data.append([
                    name,
                    comp.get("requirement", "N/A"),
                    comp.get("status", "UNKNOWN"),
                    comp.get("notes", "")
                ])
        
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 3*inch])
        
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5e35b1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3e5f5')])
        ]))
        
        return table
    
    def _build_technical_details(self, model) -> Dict:
        """Build technical details section"""
        return {
            "model_architecture": model.model_type,
            "task_type": model.task_type,
            "training_samples": model.training_samples,
            "test_samples": model.test_samples,
            "accuracy": model.accuracy,
            "target_variable": model.target_column,
            "protected_attributes": model.sensitive_columns,
            "mlflow_run_id": model.mlflow_run_id
        }
    
    def _create_technical_table(self, tech_details: Dict) -> Table:
        """Create technical details table"""
        
        data = [
            ["Model Architecture:", tech_details["model_architecture"]],
            ["Task Type:", tech_details["task_type"]],
            ["Training Samples:", f"{tech_details['training_samples']:,}"],
            ["Test Samples:", f"{tech_details['test_samples']:,}"],
            ["Accuracy:", f"{tech_details['accuracy'] * 100:.2f}%"],
            ["Target Variable:", tech_details["target_variable"]],
            ["Protected Attributes:", ", ".join(tech_details["protected_attributes"])],
            ["MLflow Run ID:", tech_details["mlflow_run_id"] or "N/A"]
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1565c0')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#90caf9'))
        ]))
        
        return table
    
    def _get_status_color(self, status: str) -> colors.Color:
        """Get color for compliance status"""
        if "COMPLIANT" in status:
            return colors.HexColor('#4caf50')
        elif "WARNING" in status:
            return colors.HexColor('#ff9800')
        elif "NON-COMPLIANT" in status or "CRITICAL" in status:
            return colors.HexColor('#f44336')
        else:
            return colors.HexColor('#9e9e9e')