import google.auth
import google.auth.transport.requests
import requests
import numpy as np
import time
import random
import os

class HeARAnalyzer:
    """
    A robust, drop-in replacement for the missing 'api_utils' in the Google Health HeAR repository.
    This implementation resolves the 'AttributeError' and mitigates 'Dependency Confusion' risks.
    """
    
    # Placeholder Endpoint Paths - These should be replaced with actual Vertex AI Endpoint IDs
    # In the official Google-Health/google-health repo, these are typically:
    # projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}
    RAW_AUDIO_ENDPOINT_PATH = os.environ.get("HEAR_RAW_AUDIO_ENDPOINT", "projects/google-health-hear/locations/us-central1/endpoints/raw-audio")
    GCS_URI_ENDPOINT_PATH = os.environ.get("HEAR_GCS_URI_ENDPOINT", "projects/google-health-hear/locations/us-central1/endpoints/gcs-uri")

    def __init__(self, credentials_path=None):
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.creds = None
        self.project = None

    def authenticate(self):
        """Refreshes and returns the Google Auth token."""
        if not self.creds:
            self.creds, self.project = google.auth.default()
        
        auth_req = google.auth.transport.requests.Request()
        self.creds.refresh(auth_req)
        return self.creds.token

    def initial_token_refresh(self, creds=None):
        """Compatibility method for the official hear_demo.ipynb."""
        if creds:
            self.creds = creds
        return self.authenticate()

    def make_prediction(self, endpoint_path, instances, gcs_bucket_name=None, gcs_creds=None):
        """
        Makes a prediction request to the HeAR Vertex AI endpoint.
        
        Args:
            endpoint_path (str): The full resource name of the endpoint.
            instances (list or np.ndarray): The audio samples or GCS URIs.
            gcs_bucket_name (str, optional): Required if using GCS URIs.
            gcs_creds (google.auth.credentials.Credentials, optional): GCS credentials.
            
        Returns:
            dict: The JSON response from the endpoint.
        """
        token = self.authenticate()
        
        # Vertex AI Prediction URL format
        # https://{LOCATION}-aiplatform.googleapis.com/v1/{ENDPOINT_PATH}:predict
        location = endpoint_path.split('/')[3]
        url = f"https://{location}-aiplatform.googleapis.com/v1/{endpoint_path}:predict"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        if isinstance(instances, np.ndarray):
            instances_list = instances.tolist()
        else:
            instances_list = instances

        # If GCS URIs are used, they might need to be prefixed with the bucket name
        if gcs_bucket_name and isinstance(instances_list[0], str):
            instances_list = [f"gs://{gcs_bucket_name}/{i}" if not i.startswith("gs://") else i for i in instances_list]

        payload = {"instances": instances_list}
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def make_prediction_with_exponential_backoff(self, endpoint_path, instances, max_retries=5):
        """Makes a prediction with exponential backoff to handle rate limits."""
        for i in range(max_retries):
            try:
                return self.make_prediction(endpoint_path, instances)
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                wait_time = (2 ** i) + random.random()
                print(f"Prediction failed, retrying in {wait_time:.2f}s... (Error: {e})")
                time.sleep(wait_time)

# Global instances for drop-in compatibility
_analyzer = HeARAnalyzer()
make_prediction = _analyzer.make_prediction
make_prediction_with_exponential_backoff = _analyzer.make_prediction_with_exponential_backoff
initial_token_refresh = _analyzer.initial_token_refresh
RAW_AUDIO_ENDPOINT_PATH = _analyzer.RAW_AUDIO_ENDPOINT_PATH
GCS_URI_ENDPOINT_PATH = _analyzer.GCS_URI_ENDPOINT_PATH
