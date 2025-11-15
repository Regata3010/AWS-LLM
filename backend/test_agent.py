# backend/scripts/test_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from dotenv import load_dotenv
from api.models.database import SessionLocal
from api.models.user import User
# --- UPDATED: Import the class directly ---
from core.ai.agent import ComplianceAgent 
from api.models.database import SQLALCHEMY_DATABASE_URL
load_dotenv()

os.environ["OPENAI_API_KEY"] = "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"
print("Using database:", SQLALCHEMY_DATABASE_URL)
async def main():
    print("BiasGuard AI Agent - Test Mode")
    print("="*50)
    
    db = SessionLocal()
    # --- UPDATED: Instantiate the class directly ---
    agent = ComplianceAgent()
    user = db.query(User).first()
    
    if not user:
        print("ERROR: No users found. Register first.")
        return
    
    print(f"User: {user.username}\n")
    
    # Test query
    test_query = "Is my model compliant with ECOA regulations?"
    print(f"Query: {test_query}\n")
    
    result = await agent.chat(
        message=test_query,
        user=user,
        db=db
    )
    
    print(f"Response:\n{result['response']}\n")
    print(f"Tools used: {result['tools_used']}")
    
    db.close()

if __name__ == "__main__":
    asyncio.run(main())