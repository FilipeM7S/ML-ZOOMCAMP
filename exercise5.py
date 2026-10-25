import json
import pickle
import re
import sklearn
from urllib.request import Request, urlopen

with open('pipeline.bin', 'rb') as f:
    pipeline = pickle.load(f)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

client = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
}

print('Q2', sklearn.__version__)

print('Q3', round(pipeline.predict_proba([lead])[0, 1], 3))

request = Request(
    'http://localhost:9696/predict',
    data=json.dumps(client).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
with urlopen(request) as response:
    result = json.load(response)
print('Q4', round(result['conversion_probability'], 3))

dockerfile = open('Dockerfile').read()
print('Q5', re.search(r'FROM (python:[^@\s]+)', dockerfile).group(1))