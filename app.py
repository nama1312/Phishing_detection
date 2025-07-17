from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib, numpy as np, pandas as pd
import tldextract, whois, re
from bs4 import BeautifulSoup
import requests
import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


base_models = {
    'rf': joblib.load("models/rf_model.joblib"),
    'lr': joblib.load("models/lr_model.joblib"),
    'dt': joblib.load("models/dt_model.joblib")
}
meta_model = joblib.load("models/meta_model.joblib")
scaler = joblib.load("models/scaler.pkl")


FEATURE_NAMES = [
    'url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url',
    'number_of_digits_in_url', 'number_of_special_char_in_url', 'number_of_hyphens_in_url',
    'number_of_underline_in_url', 'number_of_slash_in_url', 'number_of_questionmark_in_url',
    'number_of_equal_in_url', 'number_of_at_in_url', 'number_of_dollar_in_url',
    'number_of_exclamation_in_url', 'number_of_hashtag_in_url', 'number_of_percent_in_url',
    'domain_length', 'number_of_dots_in_domain', 'number_of_hyphens_in_domain',
    'having_special_characters_in_domain', 'number_of_special_characters_in_domain',
    'having_digits_in_domain', 'number_of_digits_in_domain', 'having_repeated_digits_in_domain',
    'number_of_subdomains', 'having_dot_in_subdomain', 'having_hyphen_in_subdomain',
    'average_subdomain_length', 'average_number_of_dots_in_subdomain',
    'average_number_of_hyphens_in_subdomain', 'having_special_characters_in_subdomain',
    'number_of_special_characters_in_subdomain', 'having_digits_in_subdomain',
    'number_of_digits_in_subdomain', 'having_repeated_digits_in_subdomain', 'having_path',
    'path_length', 'having_query', 'having_fragment', 'having_anchor',
    'entropy_of_url', 'entropy_of_domain'
]


def extract_features(url):
    url_length = len(url)
    number_of_dots_in_url = url.count('.')
    having_repeated_digits_in_url = 1 if re.search(r'(\d)\1{2,}', url) else 0
    number_of_digits_in_url = len(re.findall(r'\d', url))
    number_of_special_char_in_url = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url))
    number_of_hyphens_in_url = url.count('-')
    number_of_underline_in_url = url.count('_')
    number_of_slash_in_url = url.count('/')
    number_of_questionmark_in_url = url.count('?')
    number_of_equal_in_url = url.count('=')
    number_of_at_in_url = url.count('@')
    number_of_dollar_in_url = url.count('$')
    number_of_exclamation_in_url = url.count('!')
    number_of_hashtag_in_url = url.count('#')
    number_of_percent_in_url = url.count('%')

    ext = tldextract.extract(url)
    domain = ext.domain
    subdomain = ext.subdomain

    domain_length = len(domain)
    number_of_dots_in_domain = domain.count('.')
    number_of_hyphens_in_domain = domain.count('-')
    having_special_characters_in_domain = 1 if re.search(r'[!@#$%^&*(),.?":{}|<>]', domain) else 0
    number_of_special_characters_in_domain = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', domain))
    having_digits_in_domain = 1 if re.search(r'\d', domain) else 0
    number_of_digits_in_domain = len(re.findall(r'\d', domain))
    having_repeated_digits_in_domain = 1 if re.search(r'(\d)\1{2,}', domain) else 0

    number_of_subdomains = len(subdomain.split('.'))
    having_dot_in_subdomain = 1 if '.' in subdomain else 0
    having_hyphen_in_subdomain = 1 if '-' in subdomain else 0
    average_subdomain_length = np.mean([len(part) for part in subdomain.split('.')])
    average_number_of_dots_in_subdomain = number_of_dots_in_url / number_of_subdomains if number_of_subdomains > 0 else 0
    average_number_of_hyphens_in_subdomain = number_of_hyphens_in_url / number_of_subdomains if number_of_subdomains > 0 else 0
    having_special_characters_in_subdomain = 1 if re.search(r'[!@#$%^&*(),.?":{}|<>]', subdomain) else 0
    number_of_special_characters_in_subdomain = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', subdomain))
    having_digits_in_subdomain = 1 if re.search(r'\d', subdomain) else 0
    number_of_digits_in_subdomain = len(re.findall(r'\d', subdomain))
    having_repeated_digits_in_subdomain = 1 if re.search(r'(\d)\1{2,}', subdomain) else 0

    path = url.split(domain)[-1] if domain in url else ""
    having_path = 1 if path else 0
    path_length = len(path)
    having_query = 1 if '?' in url else 0
    having_fragment = 1 if '#' in url else 0
    having_anchor = 1 if '#' in url else 0

    def calculate_entropy(string):
        probabilities = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        return -sum([p * np.log2(p) for p in probabilities])

    entropy_of_url = calculate_entropy(url)
    entropy_of_domain = calculate_entropy(domain)

    return [
        url_length, number_of_dots_in_url, having_repeated_digits_in_url,
        number_of_digits_in_url, number_of_special_char_in_url, number_of_hyphens_in_url,
        number_of_underline_in_url, number_of_slash_in_url, number_of_questionmark_in_url,
        number_of_equal_in_url, number_of_at_in_url, number_of_dollar_in_url,
        number_of_exclamation_in_url, number_of_hashtag_in_url, number_of_percent_in_url,
        domain_length, number_of_dots_in_domain, number_of_hyphens_in_domain,
        having_special_characters_in_domain, number_of_special_characters_in_domain,
        having_digits_in_domain, number_of_digits_in_domain, having_repeated_digits_in_domain,
        number_of_subdomains, having_dot_in_subdomain, having_hyphen_in_subdomain,
        average_subdomain_length, average_number_of_dots_in_subdomain,
        average_number_of_hyphens_in_subdomain, having_special_characters_in_subdomain,
        number_of_special_characters_in_subdomain, having_digits_in_subdomain,
        number_of_digits_in_subdomain, having_repeated_digits_in_subdomain, having_path,
        path_length, having_query, having_fragment, having_anchor,
        entropy_of_url, entropy_of_domain
    ]


def predict_phishing(url):
    feature_vector = extract_features(url)
    features_dict = dict(zip(FEATURE_NAMES, feature_vector))
    scaled = scaler.transform([feature_vector])

    meta_input = np.zeros((1, len(base_models)))
    for i, (name, model) in enumerate(base_models.items()):
        meta_input[:, i] = model.predict_proba(scaled)[:, 1]

    final_prediction = meta_model.predict(meta_input)[0]
    probability = meta_model.predict_proba(meta_input)[0][1]

    return int(final_prediction), float(probability), features_dict


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def analyze(request: Request, url: str = Form(...)):
    prediction, prob, feats = predict_phishing(url)
    result = "🚨 Phishing" if prediction else "✅ Legitimate"
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "probability": f"{prob:.2f}",
        "features": feats,
        "url": url
    })
