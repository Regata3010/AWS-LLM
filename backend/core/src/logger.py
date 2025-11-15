import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
LOG_DIR = Path("logsnewer")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Console handler (stdout)
        logging.StreamHandler(sys.stdout),
        
        # File handler
        logging.FileHandler(
            LOG_DIR / "biasguard.log",
            mode='a',
            encoding='utf-8'
        )
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)