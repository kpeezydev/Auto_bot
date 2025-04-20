import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def load_env_vars() -> Dict[str, Any]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dictionary with environment variables
    """
    try:
        # Try to load from config/.env first
        config_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env')
        if os.path.exists(config_env_path):
            load_dotenv(config_env_path)
            logger.info(f"Loaded environment variables from {config_env_path}")
        else:
            # Try to load from root .env
            root_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            if os.path.exists(root_env_path):
                load_dotenv(root_env_path)
                logger.info(f"Loaded environment variables from {root_env_path}")
            else:
                logger.warning("No .env file found. Make sure to set environment variables manually.")
        
        # Return relevant environment variables
        return {
            'EXCHANGE_API_KEY': os.getenv('EXCHANGE_API_KEY'),
            'EXCHANGE_SECRET_KEY': os.getenv('EXCHANGE_SECRET_KEY'),
            'PAPER_TRADING': os.getenv('PAPER_TRADING', 'True').lower() in ('true', '1', 't')
        }
    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}")
        raise

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    """
    try:
        # Convert string log level to logging constant
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Configure logging to file and console
            logging.basicConfig(
                level=numeric_level,
                format=log_format,
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            logger.info(f"Logging to file: {log_file}")
        else:
            # Configure logging to console only
            logging.basicConfig(
                level=numeric_level,
                format=log_format
            )
        
        logger.info(f"Logging level set to {log_level}")
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        # Fall back to basic logging
        logging.basicConfig(level=logging.INFO, format=log_format)

# Example usage
if __name__ == "__main__":
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Load environment variables
    env_vars = load_env_vars()
    
    # Print environment variables (without sensitive data)
    print(f"Paper Trading: {env_vars['PAPER_TRADING']}")
    print(f"API Key exists: {'Yes' if env_vars['EXCHANGE_API_KEY'] else 'No'}")
    print(f"Secret Key exists: {'Yes' if env_vars['EXCHANGE_SECRET_KEY'] else 'No'}")