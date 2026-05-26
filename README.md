
# Meet & Assist Recommendation Engine

AI-based recommendation engine to suggest optimal airport assistance services.

## Features
- Trip-based scoring
- Passenger profiling
- Product ranking
- FastAPI integration

## Run the app

```bash
uvicorn app:app --reload

API Endpoint
POST /recommend

Sample Input
{
"departure": "FRA",
"arrival": "LHR",
"departure_datetime": "2026-06-02T10:00:00",
"arrival_datetime": "2026-06-02T18:00:00",
"loyalty_tier": "Gold",
"party_size": 2
}