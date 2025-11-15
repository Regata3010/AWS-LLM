from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import pandas as pd
import json
import sys
import openai
from configurations.settings import settings
from api.models.requests import ColumnSelectionRequest, ColumnOverrideRequests
from api.models.responses import ColumnSelectionResponse, ColumnOverrideResponse, TokenUsage
from api.routes.upload import get_file_from_s3
from core.src.logger import logging
from core.src.exception import CustomException
from core.validation.data_validator import DataValidator
from core.bias_detector.proxy_detector import ProxyDetector
import time
import traceback
from deprecated import deprecated

router = APIRouter()
    

# Initialize OpenAI client with proper error handling
try:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in settings")
    
    if not settings.OPENAI_API_KEY.startswith("sk-"):
        raise ValueError(f"Invalid OPENAI_API_KEY format: {settings.OPENAI_API_KEY[:10]}...")
    
    openai.api_key = settings.OPENAI_API_KEY
    client = openai
    
    
    logging.info("OpenAI client initialized successfully")
    # logging.info(f"   API key prefix: {settings.OPENAI_API_KEY[:20]}...")
    
    
except AttributeError as e:
    logging.error(f"OPENAI_API_KEY not found in settings: {e}")
    logging.error("   Check your .env file and configurations/settings.py")
    client = None
except ValueError as e:
    logging.error(f"OpenAI API key validation failed: {e}")
    client = None
except Exception as e:
    logging.error(f"OpenAI client initialization failed: {e}")
    logging.error(f"   Error type: {type(e).__name__}")
    client = None

# Log the final client state
if client is None:
    logging.warning("OpenAI client is None - LLM detection will NOT work")
    logging.warning("All requests will fall back to heuristic detection")
else:
    logging.info(f"OpenAI client ready: {type(client)}")


class EnhancedColumnSelector:
    """
    Enterprise-grade column selector with advanced LLM reasoning
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        
        
        # Pricing table
        self.pricing = {
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},
            "gpt-5-mini": {"input": 0.250, "output": 2.000},
            "gpt-5-nano": {"input": 0.050, "output": 0.400}
        }
        
        self.domain_contexts = self._load_domain_contexts()
        self.regulatory_frameworks = self._load_regulatory_frameworks()
        self.profile_cache = {}
        
        
    
    async def auto_detect(self, df: pd.DataFrame, domain: str = "finance") -> Dict:
        """Main entry point for column detection"""
        
        logging.info("="*70)
        logging.info("🔍 AUTO_DETECT CALLED")
        logging.info(f"   Client exists: {self.client is not None}")
        logging.info(f"   Model: {self.model}")
        
        try:
            if self.client:
                logging.info(" Attempting LLM detection...")
                result = await self._advanced_llm_detection(df, domain)
                logging.info(f" Success! Method: {result.get('method')}")
                return result
            else:
                logging.warning("Client is None, using heuristic")
                return self._heuristic_detection(df)
                
        except Exception as e:
            logging.error("="*70)
            logging.error("exception")
            logging.error(f"Type: {type(e).__name__}")
            logging.error(f"Message: {str(e)}")
            
            
            
    
    async def _advanced_llm_detection(self, df: pd.DataFrame, domain: str) -> Dict:
        """
        Advanced LLM-based detection with:
        - Chain-of-thought reasoning
        - Regulatory framework awareness
        - Proxy variable detection
        - Intersectionality analysis
        """
        
        cache_key = f"{df.shape}_{hash(tuple(df.columns))}"
        if cache_key in self.profile_cache:
            profile = self.profile_cache[cache_key]
            logging.info("Using cached profile")
        else:
            profile = self._build_advanced_profile(df)
            self.profile_cache[cache_key] = profile
        # Build comprehensive dataset profile
       
        
        # Get domain and regulatory context
        domain_ctx = self.domain_contexts.get(domain, self.domain_contexts["default"])
        regulatory_ctx = self.regulatory_frameworks.get(domain, self.regulatory_frameworks["default"])
        
        system_prompt = """You are Dr. Timnit Gebru, world-renowned AI Ethics researcher and algorithmic fairness expert.

Your expertise:
- Statistical fairness metrics (Disparate Impact, Demographic Parity, Equalized Odds, Calibration)
- US Civil Rights law (Title VII, ECOA, CFPB, EEOC, ADA), EU AI Act
- Proxy discrimination and intersectionality
- Domain-specific bias patterns (redlining, treatment disparities, hiring discrimination)
- Real-world case studies (COMPAS, Amazon hiring tool, Apple Card)

Analyze datasets for bias risks with regulatory audit rigor. Identify target variables, protected attributes (explicit + proxies), and intersectional risks. Provide detailed reasoning grounded in case law and research."""

        user_prompt = f"""# ANALYSIS TASK
Analyze this {domain} dataset for AI bias detection and regulatory compliance.

# DATASET PROFILE
{profile}

# DOMAIN CONTEXT
{domain_ctx}

# REGULATORY FRAMEWORK
{regulatory_ctx}

# REQUIRED OUTPUT (JSON only, no other text)
{{
  "target_variable": {{
    "column": "exact_name",
    "confidence": 0.95,
    "reasoning": {{
      "why_target": "detailed explanation with regulatory implications",
      "decision_type": "e.g., credit approval",
      "protected_decision": "yes/no - under ECOA/Title VII/etc"
    }},
    "task_type": "binary_classification|multiclass|regression"
  }},
  
  "sensitive_attributes": [
    {{
      "column": "exact_name",
      "protected_class": "race|gender|age|religion|disability|national_origin|other",
      "confidence": 0.9,
      "reasoning": {{
        "why_sensitive": "explanation with legal basis",
        "historical_discrimination": "documented evidence",
        "legal_basis": "ECOA|Title VII|ADA|etc"
      }},
      "risk_assessment": {{
        "discrimination_risk": "critical|high|moderate|low",
        "correlation_with_target": 0.4,
        "intersectional_concerns": ["combinations"]
      }},
      "mitigation_priority": 1
    }}
  ],
  
  "proxy_variables_detected": [
    {{
      "column": "name",
      "proxies_for": ["protected_class"],
      "evidence": "research/case law citation",
      "recommendation": "remove|monitor|use_with_caution"
    }}
  ],
  
  "intersectionality_analysis": {{
    "high_risk_combinations": [
      {{"attributes": ["race", "gender"], "amplification_factor": "2.3x", "evidence": "research citation"}}
    ]
  }},
  
  "recommended_fairness_metrics": ["Disparate Impact", "Demographic Parity", "Equalized Odds"],
  
  "regulatory_compliance_notes": {{
    "cfpb_requirements": "if finance",
    "eeoc_requirements": "if employment",
    "eu_ai_act": "if applicable"
  }},
  
  "column_usage_recommendations": {{
    "must_monitor": ["columns"],
    "should_remove": ["columns"],
    "use_with_caution": ["columns"],
    "safe_to_use": ["columns"]
  }}
}}

Be thorough in analysis. Cite specific regulations and case law. Consider both disparate treatment and disparate impact."""
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
        try:
            llm_start_time = time.time()
            response = self.client.ChatCompletion.create(
                model="gpt-4o-mini",  # Use latest GPT-4-mini model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency #gpt-5-nano does accept temp param.
                max_tokens=3000,  # Increased for detailed analysis #needs to be changed as per model used.
                response_format={"type": "json_object"}  # Force JSON output
            )
            llm_total_time = time.time() - llm_start_time
            actual_input_tokens = 0
            actual_output_tokens = 0
            total_tokens = 0
            
            if hasattr(response, 'usage'):
                actual_input_tokens = response.usage.prompt_tokens
                actual_output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                logging.info(f"ACTUAL tokens from OpenAI API:")
                logging.info(f"  - Input (prompt): {actual_input_tokens}")
                logging.info(f"  - Output (completion): {actual_output_tokens}")
                logging.info(f"  - Total: {total_tokens}")
                logging.info(f"  - Estimation accuracy: {estimated_tokens}/{actual_input_tokens} = {estimated_tokens/actual_input_tokens*100:.1f}%")
            else:
                # Fallback to estimates if API doesn't return usage
                logging.warning("No usage data from API, using estimates")
                actual_input_tokens = estimated_tokens
                actual_output_tokens = "Null"  
                total_tokens = actual_input_tokens + actual_output_tokens
                
            result = response.choices[0].message.content.strip()
            if "```json" in result:
                logging.info("⚠️ Removing markdown code blocks...")
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                logging.info("⚠️ Removing code blocks...")
                parts = result.split("```")
                if len(parts) >= 2:
                    result = parts[1].strip()

            # Log what we're about to parse
            logging.info(f"📝 Response to parse ({len(result)} chars):")
            logging.info(f"   Starts with: {result[:100]}")
            logging.info(f"   Ends with: {result[-100:]}")

            # Validate before parsing
            if not result:
                logging.error(" Empty response from GPT!")
                return self._heuristic_detection(df)

            if not result.startswith('{'):
                logging.error(f" Response doesn't start with '{{': {result[:200]}")
                return self._heuristic_detection(df)
            
            try:
                parsed = json.loads(result)
                logging.info(" JSON parsed successfully")
                
            except json.JSONDecodeError as e:
                logging.error("="*70)
                logging.error(f" JSON PARSE FAILED")
                logging.error(f"   Error: {e}")
                logging.error(f"   Position: line {e.lineno}, col {e.colno}")
                logging.error(f"   Full response ({len(result)} chars):")
                logging.error(result)
                logging.error("="*70)
                return self._heuristic_detection(df)
            
            
            # Validate structure
            if not self._validate_llm_response(parsed, df):
                logging.warning("LLM response validation failed, using heuristics")
                return self._heuristic_detection(df)
            
            # Extract simplified format for compatibility
            return {
                'target': parsed['target_variable']['column'],
                'target_type': parsed['target_variable']['task_type'],
                'target_confidence': parsed['target_variable']['confidence'],
                'target_rationale': parsed['target_variable']['reasoning']['why_target'],
                'sensitive': [attr['column'] for attr in parsed['sensitive_attributes']],
                'sensitive_details': parsed['sensitive_attributes'],
                'proxy_variables': parsed.get('proxy_variables_detected', []),
                'intersectionality': parsed.get('intersectionality_analysis', {}),
                'warnings': parsed.get('data_quality_warnings', []),
                'compliance_notes': parsed.get('regulatory_compliance_notes', {}),
                'recommended_metrics': parsed.get('recommended_fairness_metrics', []),
                'full_analysis': parsed,  # Include complete analysis
                'method': 'advanced_llm',
                # 'tokens_used': estimated_tokens,
                # # also keep legacy key for compatibility
                # 'estimated_tokens': estimated_tokens
                'tokens' : {
                    'input': actual_input_tokens,
                    'output': actual_output_tokens,
                    'total': total_tokens,
                    'estimated_input': estimated_tokens
                },
                'llm_time_seconds': llm_total_time
            }
            
        except json.JSONDecodeError as e:
            logging.error(f"LLM returned invalid JSON: {e}")
            return self._heuristic_detection(df)
        except Exception as e:
            logging.error(f"LLM detection failed: {e}")
            return self._heuristic_detection(df)
    
    def _build_advanced_profile(self, df: pd.DataFrame) -> str:
        """Optimized for 1500 tokens (New Version)"""
        
        lines = [
            f"# DATASET OVERVIEW",
            f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]} | Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f}MB",
            f""
        ]
        
        # Categorize columns for structured analysis
        binary_cols = []
        categorical_cols = []
        numeric_cols = []
        flagged_cols = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique = df[col].nunique()
            null_pct = df[col].isnull().sum() / len(df) * 100
            col_lower = col.lower()
            
            # Build compact column descriptor
            col_info = {
                'name': col,
                'dtype': dtype,
                'unique': unique,
                'null_pct': null_pct
            }
            
            # Add context-specific details
            if unique == 2:
                # Binary columns - likely targets or protected attributes
                values = df[col].value_counts()
                col_info['distribution'] = f"{values.iloc[0]}/{values.iloc[1]}"
                col_info['balance'] = f"{values.iloc[1]/len(df)*100:.0f}% positive"
                binary_cols.append(col_info)
            
            elif dtype in ['object', 'category']:
                # Categorical - potential protected attributes
                if unique < 20:
                    top_vals = df[col].value_counts().head(3)
                    col_info['top_values'] = list(top_vals.index)
                    col_info['top_counts'] = list(top_vals.values)
                    categorical_cols.append(col_info)
            
            elif dtype in ['int64', 'float64']:
                # Numeric columns
                col_info['range'] = f"[{df[col].min():.1f}, {df[col].max():.1f}]"
                
                # Age detection
                if ('age' in col_lower) or (0 <= df[col].min() and df[col].max() <= 120):
                    col_info['likely_age'] = True
                
                # Income detection
                if ('income' in col_lower or 'salary' in col_lower) and df[col].max() > 1000:
                    col_info['likely_income'] = True
                
                numeric_cols.append(col_info)
            
            # Flag protected attribute keywords
            protected_flags = {
                'age': 'Age (ECOA protected)',
                'gender': 'Gender (Title VII)',
                'sex': 'Sex (Title VII)',
                'race': 'Race (Title VII, ECOA)',
                'ethnic': 'Ethnicity (Title VII)',
                'religion': 'Religion (Title VII)',
                'marital': 'Marital status (ECOA)',
                'disability': 'Disability (ADA)',
                'zip': 'Geographic proxy (redlining risk)',
                'postal': 'Geographic proxy (redlining risk)',
                'nation': 'National origin (Title VII)',
                'veteran': 'Veteran status (VEVRAA)',
                'pregnancy': 'Pregnancy (Title VII)'
            }
            
            for keyword, flag_desc in protected_flags.items():
                if keyword in col_lower:
                    flagged_cols.append({
                        'name': col,
                        'flag': flag_desc,
                        'dtype': dtype,
                        'unique': unique
                    })
                    break
            
            # Flag decision keywords (target candidates)
            decision_keywords = ['approved', 'hired', 'accepted', 'rejected', 'outcome', 
                                'result', 'decision', 'status', 'default', 'churn', 'fraud']
            for keyword in decision_keywords:
                if keyword in col_lower and unique <= 10:
                    flagged_cols.append({
                        'name': col,
                        'flag': f'Decision indicator: {keyword}',
                        'dtype': dtype,
                        'unique': unique
                    })
                    break
        
        # Format output - structured and scannable
        if binary_cols:
            lines.append("## BINARY COLUMNS (Target Candidates)")
            for col in binary_cols:
                lines.append(f"• {col['name']}: {col['dtype']}, {col['distribution']} split, {col['balance']}, {col['null_pct']:.1f}% null")
        
        if categorical_cols:
            lines.append("\n## CATEGORICAL COLUMNS (Protected Attribute Candidates)")
            for col in categorical_cols:
                top_str = ', '.join([f"{v} ({c})" for v, c in zip(col['top_values'], col['top_counts'])])
                lines.append(f"• {col['name']}: {col['unique']} categories, top: {top_str}, {col['null_pct']:.1f}% null")
        
        if numeric_cols:
            lines.append("\n## NUMERIC COLUMNS")
            for col in numeric_cols:
                extra = []
                if col.get('likely_age'):
                    extra.append("Likely AGE")
                if col.get('likely_income'):
                    extra.append("Socioeconomic indicator")
                extra_str = " | ".join(extra) if extra else ""
                lines.append(f"• {col['name']}: {col['range']}, {col['unique']} unique, {col['null_pct']:.1f}% null {extra_str}")
        
        if flagged_cols:
            lines.append("\n## FLAGGED COLUMNS (Bias Risk)")
            for col in flagged_cols:
                lines.append(f"• {col['name']}: {col['flag']} | {col['dtype']}, {col['unique']} unique")
        
        # Add data quality warnings
        warnings = []
        if any(col['null_pct'] > 20 for col in binary_cols + categorical_cols + numeric_cols):
            warnings.append("High missing data (>20%) detected in some columns")
        
        if binary_cols:
            imbalanced = [col for col in binary_cols if '90%' in col['balance'] or '10%' in col['balance'] or int(col['balance'].split('%')[0]) > 80]
            if imbalanced:
                warnings.append(f"Severe class imbalance in: {', '.join([c['name'] for c in imbalanced])}")
        
        if len(flagged_cols) == 0:
            warnings.append("No obvious protected attributes detected by name - manual review needed")
        
        if warnings:
            lines.append("\n## DATA QUALITY WARNINGS")
            for warning in warnings:
                lines.append(f"• {warning}")
        
        return "\n".join(lines)
    
    def _validate_llm_response(self, parsed: Dict, df: pd.DataFrame) -> bool:
        """Validate LLM response structure and content"""
        try:
            # Check required fields
            if 'target_variable' not in parsed or 'sensitive_attributes' not in parsed:
                return False
            
            # Validate target column exists
            target_col = parsed['target_variable'].get('column')
            if target_col not in df.columns:
                logging.error(f"LLM suggested invalid target: {target_col}")
                return False
            
            # Validate sensitive columns exist
            for attr in parsed['sensitive_attributes']:
                if attr['column'] not in df.columns:
                    logging.error(f"LLM suggested invalid sensitive column: {attr['column']}")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Response validation error: {e}")
            return False
    
    def _heuristic_detection(self, df: pd.DataFrame) -> Dict:
        """Enhanced rule-based fallback with better logic"""
        
        # Target detection with better heuristics
        target_patterns = [
            'target', 'label', 'outcome', 'approved', 'hired', 'accepted',
            'income', 'class', 'result', 'decision', 'risk', 'default',
            'churn', 'fraud', 'diagnosis', 'readmit', 'salary'
        ]
        
        target = None
        target_type = 'unknown'
        target_confidence = 0.5
        
        # Priority 1: Exact name match
        exact_targets = ['target', 'label', 'outcome', 'y', 'approved', 'hired', 'accepted', 'rejected', 'default', 'churn', 'fraud']
        for col in df.columns:
            if col.lower() in exact_targets:
                target = col
                target_type = 'binary' if df[col].nunique() == 2 else 'multiclass'
                target_confidence = 0.95
                break
        
        # Priority 2: Binary columns with decision patterns
        if not target:
            decision_patterns = ['approved', 'hired', 'accepted', 'rejected', 'default', 'churn', 'outcome', 'result', 'decision']
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in decision_patterns):
                    if df[col].nunique() == 2:
                        target = col
                        target_type = 'binary'
                        target_confidence = 0.9
                        break
        
        # Priority 3: ANY binary column (before checking income/salary)
        if not target:
            for col in df.columns:
                if df[col].nunique() == 2:
                    target = col
                    target_type = 'binary'
                    target_confidence = 0.7
                    break
        
        # Priority 4: Income/salary ONLY if binary
        if not target:
            for col in df.columns:
                if df[col].nunique() == 2:
                    target = col
                    target_type = 'binary'
                    target_confidence = 0.4
                    break
        
        # Priority 5: Pattern match multiclass (ONLY if no binary found)       
        if not target:
            target_patterns = ['class', 'category', 'type', 'grade']
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in target_patterns):
                    if df[col].nunique() <= 10:
                        target = col
                        target_type = 'multiclass'
                        target_confidence = 0.6
                        break
        
        # Priority 6: Last column if binary
        if not target:
            last_col = df.columns[-1]
            if df[last_col].nunique() == 2:
                target = last_col
                target_type = 'binary'
                target_confidence = 0.5
        
        # Enhanced sensitive attribute detection
        sensitive_patterns = {
            'age': ('age', 0.9),
            'gender': ('gender', 0.9),
            'sex': ('gender', 0.9),
            'race': ('race', 0.9),
            'ethnic': ('race', 0.8),
            'religion': ('religion', 0.9),
            'marital': ('marital_status', 0.8),
            'education': ('education', 0.7),
            'zip': ('geography', 0.8),
            'postal': ('geography', 0.8),
            'nationality': ('national_origin', 0.8),
            'disability': ('disability', 0.9),
            'veteran': ('veteran_status', 0.8),
            'pregnancy': ('pregnancy', 0.9)
        }
        
        sensitive = []
        sensitive_details = []
        
        # Detect explicit protected attributes
        for col in df.columns:
            if col == target:
                continue
            
            col_lower = col.lower()
            for pattern, (protected_class, confidence) in sensitive_patterns.items():
                if pattern in col_lower:
                    sensitive.append(col)
                    sensitive_details.append({
                        'column': col,
                        'protected_class': protected_class,
                        'attribute_type': 'explicit',
                        'confidence': confidence,
                        'reasoning': {
                            'why_sensitive': f"Column name contains '{pattern}' indicating {protected_class}",
                            'legal_basis': 'Title VII Civil Rights Act' if protected_class in ['race', 'gender', 'religion', 'national_origin'] else 'Other anti-discrimination laws'
                        }
                    })
                    break
        
        # Detect proxy variables
        proxy_variables = []
        for col in df.columns:
            if col == target or col in sensitive:
                continue
            
            col_lower = col.lower()
            
            # Geographic proxies
            if any(geo in col_lower for geo in ['zip', 'postal', 'county', 'district', 'neighborhood']):
                proxy_variables.append({
                    'column': col,
                    'proxies_for': ['race', 'income'],
                    'evidence': 'Residential segregation and redlining patterns',
                    'recommendation': 'Use with extreme caution or remove'
                })
            
            # Name-based proxies
            if any(name in col_lower for name in ['first_name', 'last_name', 'firstname', 'lastname', 'name']):
                proxy_variables.append({
                    'column': col,
                    'proxies_for': ['gender', 'ethnicity'],
                    'evidence': 'Names correlate with protected characteristics',
                    'recommendation': 'Remove or anonymize'
                })
        
        # If no explicit sensitive attributes found, look for low-cardinality categoricals
        if not sensitive:
            for col in df.columns:
                if col == target:
                    continue
                if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 10:
                    sensitive.append(col)
                    sensitive_details.append({
                        'column': col,
                        'protected_class': 'unknown',
                        'attribute_type': 'inferred',
                        'confidence': 0.5,
                        'reasoning': {
                            'why_sensitive': 'Low-cardinality categorical variable, may be protected attribute',
                            'legal_basis': 'Unknown - manual review recommended'
                        }
                    })
                    if len(sensitive) >= 3:
                        break
        
        return {
            'target': target,
            'target_type': target_type,
            'target_confidence': target_confidence,
            'target_rationale': f"Heuristic detection: Pattern matching and data type analysis",
            'sensitive': sensitive,
            'sensitive_details': sensitive_details,
            'proxy_variables': proxy_variables,
            'warnings': [
                'Using rule-based detection. LLM analysis recommended for production use.',
                'Manual review of sensitive attributes strongly recommended.',
                'Check for proxy variables that may indirectly capture protected characteristics.'
            ],
            'method': 'heuristic_enhanced'
        }
    
    def _load_domain_contexts(self) -> Dict:
        """Compact domain contexts - optimized for tokens"""
        return {
            "finance": """Finance/Lending: Credit approval, loan terms, interest rates, insurance pricing | Laws: ECOA (race, gender, age, marital status, national origin), Fair Housing Act, CFPB AI/ML guidelines | Bias patterns: Redlining (zip→race), gender credit discrimination, age-based denial | High-risk proxies: zip_code→race/income, employment_history→age, credit_history→race | Compliance: 80% disparate impact rule, adverse action notices, regular audits""",

            "healthcare": """Healthcare: Treatment, diagnosis, resource allocation, insurance coverage | Laws: ACA Section 1557, HIPAA, ADA | Bias patterns: Racial pain assessment bias, gender cardiac care disparities, age treatment intensity, race-based algorithms (eGFR) | High-risk proxies: insurance_type→income, zip_code→race, language→national origin | Compliance: Clinical validation across demographics, transparency, FDA AI device review""",

            "employment": """Employment/HR: Hiring, promotion, termination, compensation, evaluation | Laws: Title VII (race, religion, sex, national origin), ADEA (age 40+), ADA, Pregnancy Act, EEOC AI guidelines, NYC Local Law 144 | Bias patterns: Resume name bias, university prestige proxy, age bias in tech, Amazon hiring tool (2018) | High-risk proxies: name→ethnicity/gender, university→socioeconomic, employment_gaps→pregnancy/age | Compliance: Four-fifths rule, validation studies, bias audits""",

            "criminal_justice": """Criminal Justice: Bail, sentencing, recidivism risk, parole | Laws: 14th Amendment Equal Protection, state algorithmic accountability, COMPAS case law | Bias patterns: COMPAS racial bias (ProPublica 2016), disparate sentencing, overpolicing feedback loops | High-risk proxies: zip_code→race, family_criminality→structural inequality | Compliance: Transparency, regular audits, human oversight""",

            "education": """Education: Admissions, financial aid, placement, disciplinary actions | Laws: Title VI (race), Title IX (sex), IDEA (disability) | Bias patterns: SAT cultural bias, disciplinary disparities, plagiarism detection bias | High-risk proxies: high_school/zip→socioeconomic, parental_education→class, test_scores→cultural bias | Compliance: Holistic review, bias testing, accommodations""",

            "default": """General: Identify protected classes under anti-discrimination laws, detect proxy variables, consider intersectionality | Laws: Title VII, ECOA, ADA, ADEA, EU AI Act | Focus: Comprehensive fairness across all protected classes"""
        }
    
    def _load_regulatory_frameworks(self) -> Dict:
        """Compact regulatory frameworks"""
        return {
            "finance": """ECOA: Prohibits credit discrimination (race, color, religion, national origin, sex, marital status, age). FCRA: Consumer credit info governance. CFPB: AI/ML model risk management. Compliance: Adverse action notices, 80% disparate impact rule, explainability, regular audits""",

            "healthcare": """Section 1557 ACA: No discrimination in health programs. ADA: Disability protections. HIPAA: Privacy/security. Compliance: Clinical validation across demographics, algorithmic transparency, FDA AI device testing""",

            "employment": """Title VII: Race, color, religion, sex, national origin. ADEA: Age 40+. ADA: Disability. EEOC AI Guidelines (2023). NYC Local Law 144: AI hiring audits. Compliance: Four-fifths rule, validation studies, accommodations""",

            "default": """US Constitution 14th Amendment, state algorithmic laws, EU AI Act, NIST AI RMF. Best practices: Document development, regular audits, human oversight, explainability"""
        }
    
    def validate_selection(self, df: pd.DataFrame, target: str, sensitive: List[str]) -> Dict:
        """Enhanced validation with regulatory considerations"""
        issues = []
        warnings = []
        recommendations = []
        
        # Validate target
        if target not in df.columns:
            issues.append(f"Target column '{target}' not found in dataset")
        else:
            # Check target properties
            unique_count = df[target].nunique()
            null_count = df[target].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            
            if unique_count > 100:
                warnings.append(f"Target has {unique_count} unique values - may need binning for classification tasks")
            
            if null_pct > 10:
                warnings.append(f"Target has {null_pct:.1f}% missing values - data quality concern")
            
            if unique_count == 2:
                # Check class balance
                value_counts = df[target].value_counts()
                minority_pct = (value_counts.min() / value_counts.sum()) * 100
                if minority_pct < 10:
                    warnings.append(f"Severe class imbalance: minority class is only {minority_pct:.1f}%")
                    recommendations.append("Consider oversampling, SMOTE, or class-weighted training")
        
        # Validate sensitive attributes
        if not sensitive or len(sensitive) == 0:
            warnings.append("No sensitive attributes selected - bias analysis will be limited")
            recommendations.append("Consider including demographics (age, gender, race) if available and relevant")
        
        for col in sensitive:
            if col not in df.columns:
                issues.append(f"Sensitive column '{col}' not found in dataset")
            elif col == target:
                issues.append(f"Target column cannot be a sensitive attribute")
            else:
                # Check group sizes
                group_sizes = df[col].value_counts()
                min_group_size = group_sizes.min()
                
                if min_group_size < 30:
                    warnings.append(f"Small sample size in '{col}' (minimum group: {min_group_size}). Statistical tests may be unreliable.")
                    recommendations.append(f"Consider collecting more data for underrepresented groups in '{col}'")
                
                if len(group_sizes) > 20:
                    warnings.append(f"'{col}' has {len(group_sizes)} groups - consider grouping into broader categories")
        
        # Check for proxy variables
        potential_proxies = []
        for col in df.columns:
            if col == target or col in sensitive:
                continue
            col_lower = col.lower()
            if any(proxy in col_lower for proxy in ['zip', 'postal', 'neighborhood', 'district']):
                potential_proxies.append(col)
        
        if potential_proxies:
            warnings.append(f"Potential proxy variables detected: {', '.join(potential_proxies)}")
            recommendations.append("Geographic variables may proxy for race/income. Consider removing or monitoring carefully.")
        
        # Statistical power check
        if target in df.columns and sensitive:
            for sens_col in sensitive:
                if sens_col in df.columns:
                    # Check if enough samples per group
                    cross_tab = pd.crosstab(df[sens_col], df[target])
                    if (cross_tab < 30).any().any():
                        warnings.append(f"Some {sens_col} × {target} combinations have <30 samples. Statistical bias tests may be underpowered.")
        
        return {
            'valid': len(issues) == 0,
            'errors': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'can_proceed': len(issues) == 0 and len(df) > 100
        }


# Initialize enhanced selector
column_selector = EnhancedColumnSelector()


@router.post("/columns/select", response_model=ColumnSelectionResponse, include_in_schema=False)
@deprecated(reason="Use BiasGuard 2.0 monitoring workflow instead")
async def select_columns(request: ColumnSelectionRequest):
    """
    Enhanced unified column selection endpoint with advanced LLM reasoning
    """
    try:
        # Get dataset
        df = await get_file_from_s3(request.file_id)
        
        if len(df) < 100:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Dataset too small for reliable analysis",
                    "current_size": len(df),
                    "minimum_required": 100,
                    "recommendation": "Upload a dataset with at least 100 rows for meaningful bias detection"
                }
            )
        start_time = time.time()
        # Build column metadata
        metadata = {}
        for col in df.columns:
            samples = df[col].dropna().head(3).tolist()[:3]
            samples = [int(x) if isinstance(x, (pd.Int64Dtype, int)) else 
                      float(x) if isinstance(x, float) else str(x) for x in samples]
            
            metadata[col] = {
                'dtype': str(df[col].dtype),
                'unique': int(df[col].nunique()),
                'missing': int(df[col].isnull().sum()),
                'missing_pct': float(df[col].isnull().sum() / len(df) * 100),
                'samples': samples
            }
       
            # Auto-detection with enhanced LLM
        if request.use_llm:
            result = await column_selector.auto_detect(df, request.domain or "finance")
        else:
            result = column_selector._heuristic_detection(df)
        proxy_results = None
    
    # Only run if we have sensitive columns detected
        if result.get('sensitive') and len(result['sensitive']) > 0:
            logging.info(f"Running proxy detection for sensitive attributes: {result['sensitive']}")
            
            proxy_start = time.time()
            proxy_detector = ProxyDetector(
                correlation_threshold=0.65,
                mutual_info_threshold=0.45,
                cramers_v_threshold=0.55,
                sample_size=5000  # Add this for optimization
            )
            
            try:
                proxy_results = proxy_detector.detect_proxies(
                    df,
                    sensitive_cols=result['sensitive'],
                    exclude_cols=[result['target']] if result.get('target') else []
                )
                
                proxy_time = time.time() - proxy_start
                logging.info(f"⏱️ Proxy detection: {proxy_time:.2f}s")
                logging.info(f"Proxy detection results: {proxy_results['summary']}")
                
                # Log critical proxies
                if proxy_results['summary']['high_risk_count'] > 0:
                    logging.warning(
                        f"HIGH-RISK PROXIES DETECTED: {proxy_results['summary']['high_risk_features']}"
                    )
                
            except Exception as e:
                logging.error(f"Proxy detection failed: {e}")
                proxy_results = {
                    "proxies": {},
                    "summary": {
                        "total_proxies_detected": 0,
                        "error": str(e),
                        "recommendations": ["Proxy detection failed - manual review recommended"]
                    }
                }
        # Validate suggestions
        validation = column_selector.validate_selection(
            df,
            result['target'],
            result['sensitive']
        )
        
        combined_warnings = result.get('warnings', [])
    
    # Add proxy warnings
        if proxy_results and proxy_results['summary']['recommendations']:
            combined_warnings.extend(proxy_results['summary']['recommendations'])
        
        # Merge with existing proxy_variables field (from LLM)
        existing_proxies = result.get('proxy_variables', [])
        detected_proxies = proxy_results['proxies'] if proxy_results else {}
        
        # Convert detected proxies to list format
        proxy_list = existing_proxies.copy()
        for feature, info in detected_proxies.items():
            proxy_list.append({
                'column': feature,
                'proxies_for': [info['proxy_for']],
                'score': info['score'],
                'method': info['method'],
                'risk_level': info['risk_level'],
                'evidence': info.get('explanation', 'Statistical association detected'),
                'recommendation': info.get('recommendation', 'Monitor carefully')
        })
            
        #adding a data validator block as well sa dropping suggested proxies    
        analysis_time = time.time() - start_time
        logging.info(f"Column selection analysis completed in {analysis_time:.2f} seconds")
        
        token_usage = None
        if result.get('tokens'):
            tokens = result['tokens']
            
            model_pricing = column_selector.pricing.get(
                column_selector.model,
                column_selector.pricing["gpt-4o-mini"]  # Default to nano
            )
            
            # GPT-4o-mini pricing (as of 2024)
            # Input: $0.150 per 1M tokens
            # Output: $0.600 per 1M tokens
            input_cost = (tokens['input'] / 1_000_000) * model_pricing['input']
            output_cost = (tokens['output'] / 1_000_000) * model_pricing['output']
            total_cost = input_cost + output_cost
            
            token_usage = {
                'input': tokens['input'],
                'output': tokens['output'],
                'total': tokens['total'],
                'estimated_input': tokens.get('estimated_input'),
                'cost_usd': round(total_cost, 6),
                'model' : column_selector.model
            }
            
            logging.info(f"Token cost: ${total_cost:.6f}")
        
        return ColumnSelectionResponse(
            file_id=request.file_id,
            all_columns=df.columns.tolist(),
            column_metadata=metadata,
            analysis_time_seconds=result.get('llm_time_seconds',0),
            suggested_target=result['target'],
            suggested_sensitive=result['sensitive'],
            target_confidence=result.get('target_confidence', 0.5),
            target_rationale=result.get('target_rationale', ''),
            sensitive_details=result.get('sensitive_details', []),
            proxy_variables=proxy_list,
            proxy_detection_summary=proxy_results['summary'] if proxy_results else None,
            intersectionality_analysis=result.get('intersectionality', {}),
            # llm_warnings=result.get('warnings', []),
            llm_warnings=combined_warnings,
            compliance_notes=result.get('compliance_notes', {}),
            recommended_metrics=result.get('recommended_fairness_metrics', []),
            full_llm_analysis=result.get('full_analysis'),
            validation=validation,
            detection_method=result.get('method', 'heuristic'),
            # prefer explicit tokens_used then fallback to legacy estimated_tokens
            tokens = token_usage,
            ready_for_training=False  # User must confirm
        )
    
    except Exception as e:
        logging.error(f"Column selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/columns/override", response_model=ColumnOverrideResponse, include_in_schema=False)
@deprecated(reason="Use BiasGuard 2.0 monitoring workflow instead")
async def override_columns(request: ColumnOverrideRequests):
    """
    User overrides LLM suggestions with their own column choices
    
    Flow:
    1. User sees LLM suggestions from /columns/select
    2. User disagrees and manually selects target + sensitive columns
    3. User calls this endpoint with their selections
    4. Backend validates and returns confirmation
    """
    try:
        # Get dataset
        df = await get_file_from_s3(request.file_id)
        
        # Build column metadata
        metadata = {}
        for col in df.columns:
            samples = df[col].dropna().head(3).tolist()[:3]
            samples = [int(x) if isinstance(x, (pd.Int64Dtype, int)) else 
                      float(x) if isinstance(x, float) else str(x) for x in samples]
            
            metadata[col] = {
                'dtype': str(df[col].dtype),
                'unique': int(df[col].nunique()),
                'missing': int(df[col].isnull().sum()),
                'missing_pct': float(df[col].isnull().sum() / len(df) * 100),
                'samples': samples
            }
        
        # Validate user's manual selection
        validation = column_selector.validate_selection(
            df,
            request.target_column,
            request.sensitive_columns
        )
        
        # Return validated user selection
        return ColumnOverrideResponse(
            file_id=request.file_id,
            target_column=request.target_column,
            sensitive_columns=request.sensitive_columns,
            validation=validation,
            ready_for_analysis=validation['valid']
            
        )
        
    except Exception as e:
        logging.error(f"Column override error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
