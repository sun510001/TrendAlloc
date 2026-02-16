import json
import os
from typing import List, Dict, Any

class AssetConfigManager:
    """
    Manages the loading and saving of asset configurations.
    
    This class handles persistent storage of asset metadata in a JSON format,
    ensuring consistent access across the data pipeline and API layers.
    """

    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ASSETS_CONFIG_PATH: str = os.path.join(PROJECT_ROOT, "config", "assets.json")

    @classmethod
    def load_assets(cls) -> List[Dict[str, Any]]:
        """
        Load asset configurations from the persistent JSON file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an asset configuration.
                                 Returns an empty list if the file does not exist or is invalid.
        """
        if not os.path.exists(cls.ASSETS_CONFIG_PATH):
            return []

        try:
            with open(cls.ASSETS_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Return an empty list if the file is malformed or unreadable
            return []

    @classmethod
    def save_assets(cls, assets: List[Dict[str, Any]]) -> None:
        """
        Save the provided asset configurations to the persistent JSON file.

        Args:
            assets (List[Dict[str, Any]]): The list of asset configurations to persist.
        """
        os.makedirs(os.path.dirname(cls.ASSETS_CONFIG_PATH), exist_ok=True)
        try:
            with open(cls.ASSETS_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(assets, f, indent=2, ensure_ascii=False)
        except IOError as e:
            # Re-importing logger locally to avoid potential circular dependencies if any
            from logger import logger
            logger.error(f"Failed to save asset configuration: {e}")
