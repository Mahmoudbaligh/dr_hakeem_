import os
import io
import google.generativeai as genai
from PIL import Image

# Load environment variables from .env if present
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()

# Initialize Gemini SDK
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Use gemini-flash-latest for fast and cost-effective image analysis
model = genai.GenerativeModel('gemini-flash-latest')

def check_image_with_gemini(image_bytes: bytes) -> str:
    """
    Analyzes the image using Gemini and returns:
    - 'NOT_SKIN' if the image is not a skin image.
    - 'HEALTHY' if the skin is completely healthy.
    - 'DISEASE' if there's any abnormality, disease, or if an error occurs.
    """
    try:
        # Load image for Gemini
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt = (
            "Analyze this image carefully. "
            "If it is NOT a human skin image at all, reply exactly with the word 'NOT_SKIN'. "
            "If it is a skin image and appears completely healthy without any skin diseases, lesions, or abnormal conditions, reply exactly with the word 'HEALTHY'. "
            "If it appears to have any skin disease, lesion, or abnormality, reply exactly with the word 'DISEASE'. "
            "Do not output anything else."
        )
        
        response = model.generate_content([prompt, img])
        
        result = response.text.strip().upper()
        
        if "NOT_SKIN" in result or "NOT SKIN" in result:
            return "NOT_SKIN"
        elif "HEALTHY" in result:
            return "HEALTHY"
        elif "DISEASE" in result:
            return "DISEASE"
        
        # Fallback to disease if the model output is not what we expect
        return "DISEASE"
        
    except Exception as e:
        print(f"[ERROR] Gemini SDK Error: {e}")
        # On error (e.g., token limit, API issue), route to custom model
        return "ERROR"
