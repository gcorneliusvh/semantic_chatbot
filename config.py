import os
import configparser
from dotenv import load_dotenv

class AppConfig:
    """A singleton class to manage application configuration."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Loads credentials from .env and looker.ini files."""
        # Find the project root by looking for the .gitignore file
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Load .env file from the root
        load_dotenv(os.path.join(project_root, '.env'))
        
        # Load looker.ini from the root
        self.looker_ini_path = os.path.join(project_root, 'looker.ini')
        if not os.path.exists(self.looker_ini_path):
            raise FileNotFoundError("looker.ini not found in the project root. Please ensure it is in the correct location.")
            
        # Set environment variables from looker.ini for SDK compatibility
        config = configparser.ConfigParser()
        config.read(self.looker_ini_path)
        
        if 'Looker' in config:
            for key, value in config['Looker'].items():
                env_key = f"LOOKERSDK_{key.upper()}"
                os.environ[env_key] = value

        print("✅ Configuration and credentials loaded successfully.")

    def get_looker_ini_path(self):
        return self.looker_ini_path

# Initialize the config singleton when the module is first imported
config = AppConfig()