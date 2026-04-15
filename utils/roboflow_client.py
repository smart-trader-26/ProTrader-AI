"""
Client for Roboflow Inference API.
Direct HTTP implementation to avoid SDK compatibility issues.
"""

import requests
import base64
import os
from typing import Dict, Any, Union, Optional

class RoboflowClient:
    """
    Lightweight client for Roboflow API.
    Handles image encoding and workflow execution.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize client.
        
        Args:
            api_key: Roboflow API Key
        """
        self.api_key = api_key
        self.base_url = "https://detect.roboflow.com"
        
    def run_workflow(self, workspace: str, workflow_id: str, images: Dict[str, Union[str, bytes]], use_cache: bool = True) -> Dict[str, Any]:
        """
        Run a Roboflow workflow on provided images.
        
        Args:
            workspace: Workspace name
            workflow_id: Workflow ID
            images: Dictionary mapping input name to image (path string or bytes)
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary containing workflow results
        """
        url = f"{self.base_url}/infer/workflows/{workspace}/{workflow_id}"
        
        # Prepare inputs
        inputs = {}
        
        for name, image_data in images.items():
            encoded_image = self._encode_image(image_data)
            inputs[name] = {"type": "base64", "value": encoded_image}
            
        payload = {
            "inputs": inputs,
            "api_key": self.api_key
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Roboflow API Error: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return {"error": str(e)}

    def _encode_image(self, image_data: Union[str, bytes]) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_data: File path or bytes
            
        Returns:
            Base64 encoded string
        """
        if isinstance(image_data, str):
            # Check if it's a file path
            if os.path.exists(image_data):
                with open(image_data, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            else:
                # Assume it's already a base64 string or url, return as is (but better to be safe)
                return image_data
        elif isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode("utf-8")
        else:
            raise ValueError("Image data must be file path or bytes")
