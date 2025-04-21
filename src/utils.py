import os
import logging
from typing import Dict, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

def load_env_vars() -> Dict[str, Any]:
    """
    Load environment variables directly from the system environment.
    
    Returns:
        Dictionary with environment variables
    """
    try:
        # Return relevant environment variables directly from system environment
        api_key = os.getenv('PIONEX_API_KEY')
        api_secret = os.getenv('PIONEX_API_SECRET')
        paper_trading = os.getenv('PAPER_TRADING', 'True').lower() in ('true', '1', 't')
        
        if not api_key or not api_secret:
            logger.warning("PIONEX_API_KEY or PIONEX_API_SECRET not found in environment variables.")
            # Depending on the application logic, you might want to raise an error here
            # or handle the missing keys appropriately elsewhere.

        return {
            'PIONEX_API_KEY': api_key,
            'PIONEX_API_SECRET': api_secret,
            'PAPER_TRADING': paper_trading
        }
    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}")
        # It's often better to return an empty dict or None and handle it upstream
        # rather than raising an exception here, unless it's critical.
        return {} # Or raise, depending on desired behavior

def setup_logging(log_level: Union[str, int] = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) as string or int
        log_file: Path to log file (optional)
    """
    # Define log format at the beginning to ensure it's available in exception handling
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    try:
        # Handle log_level as either string or int
        numeric_level = log_level
        if isinstance(log_level, str):
            numeric_level = getattr(logging, log_level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level string: {log_level}")
        elif not isinstance(log_level, int):
            raise ValueError(f"Log level must be string or int, got {type(log_level)}")
        
        # Configure logging
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
        
        logger.info(f"Logging level set to {numeric_level}")
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        # Fall back to basic logging
        logging.basicConfig(level=logging.INFO, format=log_format)

# Example usage