from sqlalchemy import text
from api.models.database import engine

print("Migrating database schema...")

with engine.connect() as conn:
    # Add organization_id column
    try:
        conn.execute(text("ALTER TABLE models ADD COLUMN organization_id TEXT"))
        print(" Added organization_id column to models table")
    except Exception as e:
        if "duplicate column" in str(e).lower():
            print("  organization_id already exists")
        else:
            print(f" Error adding organization_id: {e}")
    
    # Add created_by column
    try:
        conn.execute(text("ALTER TABLE models ADD COLUMN created_by TEXT"))
        print(" Added created_by column to models table")
    except Exception as e:
        if "duplicate column" in str(e).lower():
            print("  created_by already exists")
        else:
            print(f" Error adding created_by: {e}")
    
    # Add organization_id to bias_analyses
    try:
        conn.execute(text("ALTER TABLE bias_analyses ADD COLUMN organization_id TEXT"))
        print("Added organization_id column to bias_analyses table")
    except Exception as e:
        if "duplicate column" in str(e).lower():
            print(" organization_id already exists in bias_analyses")
        else:
            print(f"Error: {e}")
    
    conn.commit()

print("\nDatabase migration complete!")
print("You can now train models with auth fields.")