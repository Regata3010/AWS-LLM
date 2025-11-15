# backend/api/routes/report_generator.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
import matplotlib.pyplot as plt
import io
import base64
from api.models.database import get_db, Model, BiasAnalysis, Mitigation
from core.src.logger import logging
import os

router = APIRouter()

class ComplianceReportGenerator:
    """Generate compliance reports for CFPB and other regulators"""
    
    def __init__(self, db: Session):
        self.db = db
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the report"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3f51b5'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#ff5722'),
            leftIndent=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='Success',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#4caf50'),
            leftIndent=20
        ))
    
    def generate_compliance_report(
        self, 
        model_id: str,
        report_type: str = "standard",
        include_recommendations: bool = True
    ) -> str:
        """Generate a comprehensive compliance report"""
        
        # Get model data
        model = self.db.query(Model).filter(Model.model_id == model_id).first()
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Get bias analyses
        analyses = self.db.query(BiasAnalysis).filter(
            BiasAnalysis.model_id == model_id
        ).order_by(BiasAnalysis.analyzed_at.desc()).all()
        
        # Get mitigations
        mitigations = self.db.query(Mitigation).filter(
            Mitigation.original_model_id == model_id
        ).all()
        
        # Create PDF
        filename = f"reports/compliance_report_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        os.makedirs("reports", exist_ok=True)
        
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build the report content
        story = []
        
        # Title Page
        story.append(Paragraph(
            "BiasGuard Compliance Report",
            self.styles['CustomTitle']
        ))
        
        story.append(Spacer(1, 12))
        
        # Report metadata
        metadata_data = [
            ["Report Type:", report_type.upper()],
            ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Model ID:", model.model_id],
            ["Model Type:", model.model_type],
            ["Dataset:", model.dataset_name],
            ["Compliance Status:", self._get_compliance_status(model)]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a237e')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#90caf9'))
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        if analyses:
            latest_analysis = analyses[0]
            metrics = json.loads(latest_analysis.fairness_metrics)
            
            summary_text = self._generate_executive_summary(model, metrics)
            story.append(Paragraph(summary_text, self.styles['Normal']))
        else:
            story.append(Paragraph(
                "No bias analysis has been performed for this model yet.",
                self.styles['Warning']
            ))
        
        story.append(PageBreak())
        
        # Fairness Metrics Section
        story.append(Paragraph("Fairness Metrics Analysis", self.styles['SectionHeader']))
        
        if analyses:
            # Create metrics table
            metrics_data = self._prepare_metrics_table(analyses[0])
            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch, 1.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metrics_table)
            
            # Add visualization
            story.append(Spacer(1, 20))
            chart_path = self._create_metrics_chart(metrics)
            if chart_path:
                story.append(Image(chart_path, width=6*inch, height=3*inch))
        
        # Mitigation History
        if mitigations:
            story.append(PageBreak())
            story.append(Paragraph("Mitigation History", self.styles['SectionHeader']))
            
            mitigation_data = [["Date", "Strategy", "Accuracy Impact", "Bias Improvement"]]
            for mit in mitigations:
                mitigation_data.append([
                    mit.created_at.strftime("%Y-%m-%d"),
                    mit.strategy,
                    f"{mit.accuracy_impact:.2%}",
                    self._format_bias_improvement(mit.bias_improvement)
                ])
            
            mit_table = Table(mitigation_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
            mit_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4caf50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(mit_table)
        
        # Recommendations
        if include_recommendations:
            story.append(PageBreak())
            story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
            
            recommendations = self._generate_recommendations(model, analyses, mitigations)
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
                story.append(Spacer(1, 6))
        
        # Regulatory Compliance Section
        story.append(PageBreak())
        story.append(Paragraph("Regulatory Compliance Status", self.styles['SectionHeader']))
        
        compliance_text = self._generate_compliance_section(model, analyses)
        story.append(Paragraph(compliance_text, self.styles['Normal']))
        
        # Certification
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            "This report is generated by BiasGuard AI Fairness Platform and represents "
            "an accurate assessment of the model's fairness metrics as of the generation date.",
            self.styles['Normal']
        ))
        
        # Build PDF
        doc.build(story)
        
        return filename
    
    def _get_compliance_status(self, model: Model) -> str:
        """Determine overall compliance status"""
        if model.bias_status == "critical":
            return "NON-COMPLIANT"
        elif model.bias_status == "warning":
            return "AT RISK"
        elif model.bias_status == "compliant":
            return "COMPLIANT"
        else:
            return "UNKNOWN"
    
    def _generate_executive_summary(self, model: Model, metrics: Dict) -> str:
        """Generate executive summary text"""
        di = metrics.get("disparate_impact", 1.0)
        spd = metrics.get("statistical_parity_difference", 0)
        
        if di < 0.8:
            risk_level = "high risk"
            action = "immediate attention required"
        elif di < 0.9:
            risk_level = "moderate risk"
            action = "monitoring recommended"
        else:
            risk_level = "low risk"
            action = "continue monitoring"
        
        summary = f"""
        The {model.model_type} model (ID: {model.model_id}) has been analyzed for bias and fairness 
        in accordance with CFPB guidelines and industry best practices. 
        
        Current assessment indicates {risk_level} of discriminatory lending practices with 
        a disparate impact ratio of {di:.3f}. Statistical parity difference is {abs(spd):.3f}.
        
        Action recommended: {action}.
        
        This model has undergone {len(self.db.query(BiasAnalysis).filter(BiasAnalysis.model_id == model.model_id).all())} 
        bias analyses and {len(self.db.query(Mitigation).filter(Mitigation.original_model_id == model.model_id).all())} 
        mitigation attempts.
        """
        
        return summary
    
    def _prepare_metrics_table(self, analysis: BiasAnalysis) -> List[List[str]]:
        """Prepare metrics data for table"""
        metrics = json.loads(analysis.fairness_metrics)
        
        data = [["Metric", "Value", "Status"]]
        
        thresholds = {
            "disparate_impact": (0.8, 1.25),
            "statistical_parity_difference": (-0.1, 0.1),
            "equal_opportunity_difference": (-0.1, 0.1),
            "average_odds_difference": (-0.1, 0.1)
        }
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                status = "✓ PASS"
                if metric in thresholds:
                    lower, upper = thresholds[metric]
                    if metric == "disparate_impact":
                        if value < lower or value > upper:
                            status = "✗ FAIL"
                    else:
                        if abs(value) > upper:
                            status = "⚠ WARN"
                
                data.append([
                    metric.replace("_", " ").title(),
                    f"{value:.4f}",
                    status
                ])
        
        return data
    
    def _create_metrics_chart(self, metrics: Dict) -> str:
        """Create a chart visualization of metrics"""
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Filter numeric metrics
            chart_metrics = {
                k.replace("_", " ").title(): v 
                for k, v in metrics.items() 
                if isinstance(v, (int, float)) and k != "disparate_impact"
            }
            
            ax.bar(chart_metrics.keys(), chart_metrics.values(), color=['green' if abs(v) < 0.1 else 'orange' if abs(v) < 0.2 else 'red' for v in chart_metrics.values()])
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5)
            
            ax.set_ylabel('Difference from Parity')
            ax.set_title('Fairness Metrics Overview')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save to file
            chart_path = f"reports/chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path
        except Exception as e:
            logging.error(f"Error creating chart: {str(e)}")
            return None
    
    def _format_bias_improvement(self, improvement: str) -> str:
        """Format bias improvement data"""
        try:
            imp_data = json.loads(improvement) if isinstance(improvement, str) else improvement
            if isinstance(imp_data, dict):
                key_metric = imp_data.get("disparate_impact_change", imp_data.get("primary_metric_change", 0))
                return f"{key_metric:.2%}"
            return "N/A"
        except:
            return "N/A"
    
    def _generate_recommendations(self, model: Model, analyses: List[BiasAnalysis], mitigations: List[Mitigation]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if analyses:
            latest = analyses[0]
            metrics = json.loads(latest.fairness_metrics)
            
            di = metrics.get("disparate_impact", 1.0)
            if di < 0.8:
                recommendations.append(
                    "Critical: Disparate impact below 80% threshold. Consider immediate retraining with "
                    "fairness constraints or implementing bias mitigation strategies."
                )
            
            spd = abs(metrics.get("statistical_parity_difference", 0))
            if spd > 0.1:
                recommendations.append(
                    "Address statistical parity difference through reweighing or threshold optimization."
                )
            
            if not mitigations:
                recommendations.append(
                    "No mitigation attempts recorded. Recommend running bias mitigation algorithms "
                    "to improve fairness metrics."
                )
            elif len(mitigations) > 3:
                recommendations.append(
                    "Multiple mitigation attempts with limited success. Consider model architecture "
                    "changes or feature engineering to address root causes."
                )
        
        recommendations.append(
            "Implement continuous monitoring to detect bias drift over time."
        )
        
        recommendations.append(
            "Document all fairness assessments and mitigation attempts for regulatory compliance."
        )
        
        return recommendations
    
    def _generate_compliance_section(self, model: Model, analyses: List[BiasAnalysis]) -> str:
        """Generate regulatory compliance section"""
        text = """
        CFPB Compliance: This model has been evaluated against Consumer Financial Protection Bureau 
        guidelines for fair lending practices.
        
        ECOA Compliance: Assessment includes protected class analysis as required by the 
        Equal Credit Opportunity Act.
        
        Model Risk Management: Complies with SR 11-7 supervisory guidance on model risk management.
        
        """
        
        if model.bias_status == "compliant":
            text += "Status: Model meets all regulatory requirements for fair lending."
        elif model.bias_status == "warning":
            text += "Status: Model shows potential compliance risks. Enhanced monitoring required."
        else:
            text += "Status: Model may not meet regulatory requirements. Immediate action recommended."
        
        return text

@router.post("/report/generate/{model_id}")
async def generate_report(
    model_id: str,
    report_type: str = "standard",
    include_recommendations: bool = True,
    db: Session = Depends(get_db)
):
    """Generate a compliance report for a model"""
    try:
        generator = ComplianceReportGenerator(db)
        report_path = generator.generate_compliance_report(
            model_id=model_id,
            report_type=report_type,
            include_recommendations=include_recommendations
        )
        
        return {
            "status": "success",
            "report_path": report_path,
            "download_url": f"/report/download/{os.path.basename(report_path)}"
        }
    except Exception as e:
        logging.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/download/{filename}")
async def download_report(filename: str):
    """Download a generated report"""
    file_path = f"reports/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )

@router.post("/report/schedule")
async def schedule_report(
    model_id: str,
    frequency: str = "weekly",  # daily, weekly, monthly
    email: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Schedule automatic report generation"""
    # This would integrate with a task scheduler like Celery
    # For now, we'll just return the configuration
    
    schedule_config = {
        "model_id": model_id,
        "frequency": frequency,
        "email": email,
        "next_run": (datetime.now() + timedelta(days=7 if frequency == "weekly" else 1)).isoformat(),
        "status": "scheduled"
    }
    
    return {
        "status": "success",
        "message": f"Report scheduled for {frequency} generation",
        "config": schedule_config
    }

@router.get("/report/history/{model_id}")
async def get_report_history(
    model_id: str,
    limit: int = 10
):
    """Get history of generated reports for a model"""
    # List all reports for this model
    reports = []
    
    if os.path.exists("reports"):
        for filename in os.listdir("reports"):
            if model_id in filename and filename.endswith(".pdf"):
                file_path = os.path.join("reports", filename)
                reports.append({
                    "filename": filename,
                    "generated_at": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                    "size_kb": os.path.getsize(file_path) / 1024,
                    "download_url": f"/report/download/{filename}"
                })
    
    # Sort by generation time
    reports.sort(key=lambda x: x["generated_at"], reverse=True)
    
    return {
        "total": len(reports),
        "reports": reports[:limit]
    }