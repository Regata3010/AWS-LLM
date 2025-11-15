import os, sys
import pandas as pd
import openai
import json
from core.src.exception import CustomException
from dotenv import load_dotenv
from typing import Dict, List
import streamlit as st

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_csv(file_path: str):
    """Loads CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e)


def llm_column_guess(df: pd.DataFrame):
    """
    Enhanced LLM column detection with better JSON schema and reasoning.
    """
    try:
        sample_data = df.head(3).to_csv(index=False)
        col_info = []
        
        # Simple column analysis for LLM context
        for col in df.columns:
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            dtype = str(df[col].dtype)
            sample_vals = df[col].dropna().head(2).tolist()
            
            col_info.append(f"{col}: {dtype}, {unique_count} unique, {missing_count} missing, samples: {sample_vals}")

        enhanced_prompt = f'''
You are an expert data scientist analyzing a dataset for AI bias detection.

DATASET SAMPLE:
{sample_data}

COLUMN ANALYSIS:
{chr(10).join(col_info)}

TASK: Identify target column (what we're predicting) and sensitive attributes (demographics that could cause bias).

TARGET COLUMN: Usually the outcome/result variable (income, approved, outcome, target, label, etc.)
SENSITIVE COLUMNS: Demographics like age, gender, race, education, marital_status, etc.

RESPOND WITH ONLY THIS JSON:
{{
  "target": "exact_column_name",
  "sensitive": ["column1", "column2"],
  "confidence": {{
    "target": 0.95,
    "sensitive": 0.85
  }},
  "reasoning": {{
    "target": "Brief reason why this is target",
    "sensitive": "Brief reason why these are sensitive"
  }}
}}
'''

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert data scientist. Always respond with valid JSON only."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        
        # Clean JSON extraction
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif not result.startswith("{"):
            raise CustomException("LLM response is not valid JSON", sys)

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            raise CustomException(f"Invalid JSON from LLM: {result}", sys)

        # Validate columns exist
        if parsed['target'] not in df.columns:
            raise CustomException(f"Target column '{parsed['target']}' not found", sys)
        
        for sens_col in parsed['sensitive']:
            if sens_col not in df.columns:
                raise CustomException(f"Sensitive column '{sens_col}' not found", sys)

        return parsed['target'], parsed['sensitive']

    except Exception as e:
        raise CustomException(e, sys)


def api_column_detection(df: pd.DataFrame) -> Dict:
    """
    API-ready column detection for FastAPI endpoints.
    Returns detailed results with reasoning.
    """
    try:
        sample_data = df.head(3).to_csv(index=False)
        col_info = []
        
        # Prepare column info for LLM
        for col in df.columns:
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            dtype = str(df[col].dtype)
            sample_vals = df[col].dropna().head(2).tolist()
            
            col_info.append(f"{col}: {dtype}, {unique_count} unique, {missing_count} missing, samples: {sample_vals}")

        enhanced_prompt = f'''
You are an expert data scientist analyzing a dataset for AI bias detection.

DATASET SAMPLE:
{sample_data}

COLUMN ANALYSIS:
{chr(10).join(col_info)}

TASK: Identify target column and sensitive attributes for bias analysis.

RESPOND WITH ONLY THIS JSON:
{{
  "target": "exact_column_name",
  "sensitive": ["column1", "column2"],
  "confidence": {{
    "target": 0.95,
    "sensitive": 0.85
  }},
  "reasoning": {{
    "target": "Brief reason",
    "sensitive": "Brief reason"
  }},
  "data_quality": {{
    "target_missing": 0,
    "sensitive_missing": {{"age": 5}},
    "ready_for_analysis": true
  }}
}}
'''

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Expert data scientist. Respond with valid JSON only."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        
        parsed = json.loads(result)
        
        # Validate columns exist
        if parsed['target'] not in df.columns:
            raise ValueError(f"Target column '{parsed['target']}' not found")
        
        for sens_col in parsed['sensitive']:
            if sens_col not in df.columns:
                raise ValueError(f"Sensitive column '{sens_col}' not found")

        return {
            'success': True,
            'target_column': parsed['target'],
            'sensitive_columns': parsed['sensitive'],
            'llm_confidence': parsed.get('confidence', {}),
            'llm_reasoning': parsed.get('reasoning', {}),
            'data_quality': parsed.get('data_quality', {}),
            'method': 'enhanced_llm'
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'target_column': None,
            'sensitive_columns': []
        }

def prompt_user_for_override(default_target: str, default_sensitive: list, df_columns: list):
    """Ask user if they want to override LLM guesses."""
    st.subheader("🤖 LLM Suggested Columns")
    st.markdown(f"**Target Column:** `{default_target}`")
    st.markdown("**Sensitive Columns:**")
    st.write(default_sensitive)

    override = st.radio("❓ Do you want to override these?", ("No", "Yes"))

    if override == "Yes":
        target_col = st.selectbox("Select Target Column", df_columns, index=df_columns.index(default_target))
        sensitive_col = st.multiselect("Select Sensitive Columns", df_columns, default=default_sensitive)
    else:
        st.info("ℹ️ Using LLM-suggested columns.")
        target_col = default_target
        sensitive_col = default_sensitive
        
    st.markdown("---")
    st.subheader("✅ Final Column Selection")
    st.markdown(f"**Target Column:** `{target_col}`")
    st.markdown("**Sensitive Columns:**")
    st.write(sensitive_col)

    return target_col, sensitive_col


def final_columns_from_csv(df: pd.DataFrame, use_llm=True):
    """Final pipeline to determine selected columns from dataset"""
    try:
        if use_llm:
            target, sensitive = llm_column_guess(df)
        else:
            target = input("Enter target column name: ").strip()
            sensitive_input = input("Enter sensitive column(s), comma-separated: ").strip()
            sensitive = [col.strip() for col in sensitive_input.split(",")]

        target_col, sensitive_col = prompt_user_for_override(target, sensitive, df.columns.tolist())
        return target_col, sensitive_col

    except Exception as e:
        raise CustomException(e)