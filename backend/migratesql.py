import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.models import database, user
import json

def migrate_data(sqlite_url: str, postgres_url: str):
    """Migrate all data from SQLite to PostgreSQL"""
    
    print("🚀 Starting migration...")
    print(f" Source: {sqlite_url}")
    print(f" Target: {postgres_url}")
    
    # Create engines
    sqlite_engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    postgres_engine = create_engine(postgres_url, pool_pre_ping=True)
    
    # Create sessions
    SQLiteSession = sessionmaker(bind=sqlite_engine)
    PostgresSession = sessionmaker(bind=postgres_engine)
    
    sqlite_session = SQLiteSession()
    postgres_session = PostgresSession()
    
    try:
        # Create tables in PostgreSQL
        print("\n📊 Creating PostgreSQL tables...")
        database.Base.metadata.create_all(postgres_engine)
        user.Base.metadata.create_all(postgres_engine)
        print("✅ Tables created")
        
        # Migrate Organizations
        print("\n👥 Migrating organizations...")
        orgs = sqlite_session.query(user.Organization).all()
        for org in orgs:
            postgres_session.merge(org)
        postgres_session.commit()
        print(f"✅ Migrated {len(orgs)} organizations")
        
        # Migrate Users
        print("\n🔐 Migrating users...")
        users = sqlite_session.query(user.User).all()
        for u in users:
            postgres_session.merge(u)
        postgres_session.commit()
        print(f"✅ Migrated {len(users)} users")
        
        # Migrate Models
        print("\n🤖 Migrating models...")
        models = sqlite_session.query(database.Model).all()
        for model in models:
            postgres_session.merge(model)
        postgres_session.commit()
        print(f"✅ Migrated {len(models)} models")
        
        # Migrate External Models
        print("\n📡 Migrating external models...")
        ext_models = sqlite_session.query(database.ExternalModel).all()
        for model in ext_models:
            postgres_session.merge(model)
        postgres_session.commit()
        print(f"✅ Migrated {len(ext_models)} external models")
        
        # Migrate Bias Analyses
        print("\n📈 Migrating bias analyses...")
        analyses = sqlite_session.query(database.BiasAnalysis).all()
        for analysis in analyses:
            postgres_session.merge(analysis)
        postgres_session.commit()
        print(f"✅ Migrated {len(analyses)} analyses")
        
        # Migrate Mitigations
        print("\n🛡️ Migrating mitigations...")
        mitigations = sqlite_session.query(database.Mitigation).all()
        for mitigation in mitigations:
            postgres_session.merge(mitigation)
        postgres_session.commit()
        print(f"✅ Migrated {len(mitigations)} mitigations")
        
        # Migrate Prediction Logs
        print("\n📝 Migrating prediction logs...")
        logs = sqlite_session.query(database.PredictionLog).all()
        for log in logs:
            postgres_session.merge(log)
        postgres_session.commit()
        print(f"✅ Migrated {len(logs)} prediction logs")
        
        # Migrate Invitations
        print("\n✉️ Migrating invitations...")
        invites = sqlite_session.query(user.Invitation).all()
        for invite in invites:
            postgres_session.merge(invite)
        postgres_session.commit()
        print(f"✅ Migrated {len(invites)} invitations")
        
        print("\n🎉 Migration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        postgres_session.rollback()
        raise
    finally:
        sqlite_session.close()
        postgres_session.close()

def export_sqlite_to_json(sqlite_url: str, output_file: str = "backup.json"):
    """Export SQLite data to JSON for backup"""
    
    print(f"💾 Exporting SQLite to {output_file}...")
    
    engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    session = Session()
    
    backup = {
        "organizations": [],
        "users": [],
        "models": [],
        "external_models": [],
        "bias_analyses": [],
        "mitigations": [],
        "prediction_logs": [],
        "invitations": []
    }
    
    try:
        # Export each table
        for org in session.query(user.Organization).all():
            backup["organizations"].append({
                "id": org.id,
                "name": org.name,
                "plan_tier": org.plan_tier,
                "max_users": org.max_users,
                "max_models": org.max_models,
                "created_at": org.created_at.isoformat()
            })
        
        for u in session.query(user.User).all():
            backup["users"].append({
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "hashed_password": u.hashed_password,
                "organization_id": u.organization_id,
                "role": u.role,
                "is_superuser": u.is_superuser,
                "created_at": u.created_at.isoformat()
            })
        
        for model in session.query(database.Model).all():
            backup["models"].append({
                "model_id": model.model_id,
                "model_type": model.model_type,
                "task_type": model.task_type,
                "accuracy": model.accuracy,
                "bias_status": model.bias_status,
                "created_at": model.created_at.isoformat()
            })
        
        with open(output_file, 'w') as f:
            json.dump(backup, f, indent=2)
        
        print(f"✅ Backup saved to {output_file}")
        
    finally:
        session.close()

if __name__ == "__main__":
    # Configuration
    SQLITE_URL = "sqlite:///./backend/biasguard2.0.db"
    POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://biasguard:changeme123@localhost:5432/biasguard")
    
    print("=" * 60)
    print("🗄️  BiasGuard Database Migration")
    print("=" * 60)
    
    # Create backup first
    export_sqlite_to_json(SQLITE_URL, "biasguard_backup.json")
    
    # Run migration
    migrate_data(SQLITE_URL, POSTGRES_URL)
    
    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print("=" * 60)