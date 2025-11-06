"""
Configuration and constants for the bug reporting chatbot.
Centralized settings for easy management and deployment.
"""
import os


class Config:
    """Application configuration."""
    
    # OpenRouter API Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    # LLM Settings
    LLM_MAX_TOKENS = 500
    EXTRACTION_MAX_TOKENS = 200
    END_DETECTION_MAX_TOKENS = 100
    
    # Conversation Settings
    MAX_CONVERSATION_TURNS = 20
    
    # Bug Status Values
    VALID_STATUSES = ["Open", "In Progress", "Testing", "Resolved", "Closed"]
    
    # Data Paths
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    TRACES_DIR = "results/traces"
    OUTPUTS_DIR = "results/outputs"
    
    @classmethod
    def validate(cls) -> None:
        """Validate that all required configuration is set."""
        if not cls.OPENROUTER_API_KEY or not cls.OPENROUTER_MODEL:
            raise ValueError("OPENROUTER_API_KEY and OPENROUTER_MODEL must be set in environment")
    
    @classmethod
    def get_api_client_config(cls) -> dict:
        """Get configuration for OpenAI API client."""
        return {
            "api_key": cls.OPENROUTER_API_KEY,
            "base_url": cls.OPENROUTER_BASE_URL
        }

