import os

import requests

GH_SECRET = os.environ.get('GH_SECRET')
GH_TRAFFIC_VIEWS_URL = "https://api.github.com/repos/explodinggradients/ragas/traffic/views"

repo_response = requests.get(GH_TRAFFIC_VIEWS_URL,headers={
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f"Bearer {GH_SECRET}"
})
print(repo_response.json())
