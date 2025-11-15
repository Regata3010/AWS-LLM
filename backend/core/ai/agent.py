# backend/core/ai/agent.py
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session
from api.models.user import User, Organization
from core.ai.tools import get_tools_for_user
from core.ai.rag import get_rag_tools
import os
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """You are BiasGuard AI.

CRITICAL INSTRUCTION: You MUST use tools for EVERY query. Do NOT answer from memory.

When user asks: "What models do I have?"
YOU MUST: Call get_user_models() tool BEFORE answering

When user asks: "What is [legal term]?"
YOU MUST: Call search_regulations() tool BEFORE answering

NEVER say you have information without calling a tool first.

Current user: {username}
Role: {role}
Organization: {organization}
Superuser: {is_superuser}"""

class ComplianceAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    async def chat(self, message: str, user: User, db: Session, thread_id: str = None):
        # Get tools
        tools = get_tools_for_user(db, user) + get_rag_tools()
        print(f"\nDEBUG: Agent has {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}")
        
        # Get org name and create system message (unchanged)
        org = db.query(Organization).filter(Organization.id == user.organization_id).first()
        org_name = org.name if org else "N/A"
        system_msg = SYSTEM_PROMPT.format(
            username=user.username,
            role=user.role,
            organization=org_name,
            is_superuser=user.is_superuser
        )
        
        # --- UPDATED AGENT CREATION ---
        agent_executor = create_agent(model=self.llm, tools=tools, system_prompt=system_msg)
        
        print("\nDEBUG: Executing agent with verbose=True...")
        result = await agent_executor.ainvoke(
            {"messages": [("user", message)]},
            config={"callbacks": []}  # Enable to see tool calls
        )
        
        tools_used = []
        for msg in result.get("messages", []):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tools_used.append(tool_call['name'])
        
        print("\nDEBUG: Raw result keys:", result.keys())
        print("\nDEBUG: Messages in result:", len(result.get("messages", [])))
        
        # Print each message type
        for i, msg in enumerate(result.get("messages", [])):
            print(f"\nMessage {i}:")
            print(f"  Type: {type(msg).__name__}")
            print(f"  Has tool_calls: {hasattr(msg, 'tool_calls')}")
            if hasattr(msg, 'tool_calls'):
                print(f"  Tool calls: {msg.tool_calls}")
        
        
        
    
        # Extract response
        final_msg = result["messages"][-1].content if result.get("messages") else ""
        
        # # --- NEW CODE: Extract Tool Names ---
        # tools_used = set()
        # for msg in result.get("messages", []):
        #     # 1. Check if the message object has a 'tool_calls' attribute and if it's not empty.
        #     if hasattr(msg, "tool_calls") and msg.tool_calls:
        #         for tool_call in msg.tool_calls:
        #             # The tool_call object usually has a .name attribute
        #             if hasattr(tool_call, 'name'):
        #                 tools_used.add(tool_call.name)
            
        # # Clean up the set into a list
        # tools_used_list = sorted(list(tools_used))
        
        return {
            "response": final_msg,
            "tools_used": list(set(tools_used)),
            "messages": result.get("messages", [])
        }