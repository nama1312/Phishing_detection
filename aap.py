import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import re
import tldextract
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Load the model and scaler
model = joblib.load('phishing_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature extraction function
def extract_features(url):
    # Example feature extraction logic
    features = {}
    extracted_url = tldextract.extract(url)

    features['url_length'] = len(url)
    features['number_of_dots_in_url'] = url.count('.')
    features['having_repeated_digits_in_url'] = len(re.findall(r'(\d)\1+', url)) > 0
    features['number_of_digits_in_url'] = sum(c.isdigit() for c in url)
    features['number_of_special_char_in_url'] = sum(c in "!@#$%^&*()_+[]{}|;:,.<>?/`~" for c in url)
    features['number_of_hyphens_in_url'] = url.count('-')
    features['number_of_underline_in_url'] = url.count('_')
    features['number_of_slash_in_url'] = url.count('/')
    features['number_of_questionmark_in_url'] = url.count('?')
    features['number_of_equal_in_url'] = url.count('=')
    features['number_of_at_in_url'] = url.count('@')
    features['number_of_dollar_in_url'] = url.count('$')
    features['number_of_exclamation_in_url'] = url.count('!')
    features['number_of_hashtag_in_url'] = url.count('#')
    features['number_of_percent_in_url'] = url.count('%')

    features['domain_length'] = len(extracted_url.domain)
    features['number_of_dots_in_domain'] = extracted_url.domain.count('.')
    features['number_of_hyphens_in_domain'] = extracted_url.domain.count('-')
    features['having_special_characters_in_domain'] = len(re.findall(r'[!@#$%^&*()_+[\]{}|;:,.<>?/]', extracted_url.domain)) > 0
    features['number_of_special_characters_in_domain'] = sum(c in "!@#$%^&*()_+[]{}|;:,.<>?/`~" for c in extracted_url.domain)
    features['having_digits_in_domain'] = any(c.isdigit() for c in extracted_url.domain)
    features['number_of_digits_in_domain'] = sum(c.isdigit() for c in extracted_url.domain)
    features['having_repeated_digits_in_domain'] = len(re.findall(r'(\d)\1+', extracted_url.domain)) > 0
    features['number_of_subdomains'] = len(extracted_url.subdomain.split('.'))
    features['having_dot_in_subdomain'] = '.' in extracted_url.subdomain
    features['having_hyphen_in_subdomain'] = '-' in extracted_url.subdomain
    features['average_subdomain_length'] = np.mean([len(sub) for sub in extracted_url.subdomain.split('.')]) if extracted_url.subdomain else 0
    features['average_number_of_dots_in_subdomain'] = features['number_of_dots_in_domain'] / (features['number_of_subdomains'] + 1)
    features['average_number_of_hyphens_in_subdomain'] = features['number_of_hyphens_in_domain'] / (features['number_of_subdomains'] + 1)
    features['having_special_characters_in_subdomain'] = len(re.findall(r'[!@#$%^&*()_+[\]{}|;:,.<>?/]', extracted_url.subdomain)) > 0
    features['number_of_special_characters_in_subdomain'] = sum(c in "!@#$%^&*()_+[]{}|;:,.<>?/`~" for c in extracted_url.subdomain)
    features['having_digits_in_subdomain'] = any(c.isdigit() for c in extracted_url.subdomain)
    features['number_of_digits_in_subdomain'] = sum(c.isdigit() for c in extracted_url.subdomain)
    features['having_repeated_digits_in_subdomain'] = len(re.findall(r'(\d)\1+', extracted_url.subdomain)) > 0
    features['having_path'] = '/' in url
    features['path_length'] = len(url.split('/', 1)[-1]) if '/' in url else 0
    features['having_query'] = '?' in url
    features['having_fragment'] = '#' in url
    features['having_anchor'] = '#' in url
    features['entropy_of_url'] = -sum(p * np.log2(p) for p in [url.count(c) / len(url) for c in set(url)]) if len(url) > 0 else 0
    features['entropy_of_domain'] = -sum(p * np.log2(p) for p in [extracted_url.domain.count(c) / len(extracted_url.domain) for c in set(extracted_url.domain)]) if len(extracted_url.domain) > 0 else 0
    
    # Convert boolean features to integers
    for key in features:
        if isinstance(features[key], bool):
            features[key] = int(features[key])
    
    return features

# Input schema
class URLInput(BaseModel):
    url: str

# Prediction endpoint
@app.post("/predict/")
def predict(url_input: URLInput):
    features = extract_features(url_input.url)
    input_df = pd.DataFrame([features])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    return {"prediction": int(prediction[0]), "phishing_probability": float(prediction_proba[0][1])}

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
