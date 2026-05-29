
# “Before introducing GenAI, the system relied purely on deterministic scoring. 
# After integrating GenAI, we enhanced the model with contextual reasoning, 
# leading to more refined and human-like recommendations.”

import pandas as pd
import json
# from openai import OpenAI
from genai import generate_explanation   # your existing

# client = OpenAI(api_key="YOUR_API_KEY")
from dotenv import load_dotenv
import os


import joblib

# ✅ Load ML artifacts
try:
    model = joblib.load("propensity_model.pkl")
    le_loyalty = joblib.load("le_loyalty.pkl")
    le_tier = joblib.load("le_tier.pkl")
    le_time = joblib.load("le_time.pkl")
except Exception:
    model = None


try:
    history = pd.read_excel("passenger_purchase_history_200.xlsx", engine="openpyxl")
except Exception:
    history = pd.DataFrame()


def is_existing_user(passenger_id):
    if not passenger_id or history.empty:
        return False

    return not history[history["PassengerID"] == passenger_id].empty



load_dotenv()

from langchain_groq import ChatGroq

# ✅ Initialize OpenAI
# client = OpenAI(api_key="YOUR_API_KEY")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# ✅ Load datasets
trips = pd.read_excel("trips_with_flight_1000.xlsx", engine="openpyxl")
products = pd.read_excel("meet_assist_products_full_15.xlsx", engine="openpyxl")

trips["Departure"] = trips["Departure"].astype(str).str.strip().str.upper()
trips["Arrival"] = trips["Arrival"].astype(str).str.strip().str.upper()

products.columns = products.columns.str.strip()
products["PriceINR"] = pd.to_numeric(products["PriceINR"], errors="coerce")
products["PartnerSLAScore"] = pd.to_numeric(products["PartnerSLAScore"], errors="coerce")
products = products.dropna()



# ✅ -------------------------------
# 🔧 INPUT VALIDATION & NORMALIZATION
# -------------------------------
def _normalize_airport_code(code):
    return str(code).strip().upper()


def _parse_datetime(value, field_name):
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Invalid {field_name}. Expected ISO datetime string.")
    return dt


def _normalize_loyalty_tier(loyalty_tier):
    loyalty = str(loyalty_tier).strip().upper()

    if loyalty in {"HON CIRCLE", "PLATINUM"}:
        return "HON Circle"
    if loyalty in {"SENATOR", "GOLD"}:
        return "Senator"
    return "Frequent Traveller"


def _validate_and_normalize_input(data):
    required_fields = [
        "departure",
        "arrival",
        "departure_datetime",
        "arrival_datetime",
        "loyalty_tier",
        "party_size",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing required input fields: {', '.join(missing)}")

    normalized = dict(data)
    normalized["departure"] = _normalize_airport_code(data.get("departure", ""))
    normalized["arrival"] = _normalize_airport_code(data.get("arrival", ""))

    if len(normalized["departure"]) != 3 or len(normalized["arrival"]) != 3:
        raise ValueError("departure and arrival should be 3-letter IATA-style airport codes.")

    try:
        party_size = int(data.get("party_size"))
    except (TypeError, ValueError):
        raise ValueError("party_size must be an integer.")
    if party_size < 1:
        raise ValueError("party_size must be >= 1.")
    normalized["party_size"] = party_size

    departure_dt = _parse_datetime(data.get("departure_datetime"), "departure_datetime")
    arrival_dt = _parse_datetime(data.get("arrival_datetime"), "arrival_datetime")
    if arrival_dt <= departure_dt:
        raise ValueError("arrival_datetime must be later than departure_datetime.")

    normalized["departure_datetime"] = departure_dt
    normalized["arrival_datetime"] = arrival_dt
    return normalized


# ✅ -------------------------------
# 🧠 GenAI SCORE ADJUSTMENT (NEW ✅)
# -------------------------------
# def apply_genai_adjustment(data, scored_products):

    try:
        top_product = sorted(
            scored_products,
            key=lambda x: x["score"],
            reverse=True
        )[:5]

        product_text = ""

        for i, item in enumerate(top_product):
            p = item["product"]
            product_text += f"""{i+1}. {p['ProductName']} (Tier: {p['ServiceTier']}, SLA: {p['PartnerSLAScore']})\n"""

        prompt = f"""
You are an AI scoring assistant.

Passenger:
- Loyalty: {data.get('loyalty_tier')}
- Party Size: {data.get('party_size')}

Task:
Assign a score between 0 and 1 for each product.

Products:
{product_text}

Return JSON format:
{{ "ProductName": score }}
"""

        response = llm.invoke(prompt)

        genai_scores = json.loads(response.choices[0].message.content)

    except Exception:
        # ✅ fallback if GenAI fails
        return scored_products

    updated = []

    for item in scored_products:

        pname = item["product"]["ProductName"]
        base_score = item["score"]

        genai_score = genai_scores.get(pname, base_score)

        # ✅ combine
        final_score = (0.7 * base_score) + (0.3 * genai_score)

        item["score"] = round(final_score, 4)

        updated.append(item)

    return updated

def apply_genai_adjustment(data, scored_products):

    try:
        top_product = sorted(
            scored_products,
            key=lambda x: x["score"],
            reverse=True
        )[:5]

        product_text = ""

        for i, item in enumerate(top_product):
            p = item["product"]

            refund = p.get("RefundPolicy", "Unknown")

            product_text += f"""{i+1}. {p['ProductName']}
Tier: {p['ServiceTier']}
SLA: {p['PartnerSLAScore']}
Refund: {refund}

"""

        prompt = f"""
You are an AI scoring assistant.

Passenger:
- Loyalty: {data.get('loyalty_tier')}
- Party Size: {data.get('party_size')}

Important:
- For high-risk or uncertain trips, prefer refundable services
- Premium users prefer better service and flexibility

Task:
Assign a score between 0 and 1 for each product.

Consider:
- SLA (service quality)
- Tier suitability
- Refund policy (VERY IMPORTANT)
- Overall passenger needs

Products:
{product_text}

Return JSON format:
{{ "ProductName": score }}
"""

        response = llm.invoke(prompt)

        genai_scores = json.loads(response.content)

    except Exception:
        return scored_products

    updated = []

    for item in scored_products:

        pname = item["product"]["ProductName"]
        base_score = item["score"]

        genai_score = genai_scores.get(pname, base_score)

        # ✅ recommended safer weight
        final_score = (0.8 * base_score) + (0.2 * genai_score)

        item["score"] = round(final_score, 4)

        updated.append(item)

    return updated


# ✅ -------------------------------
# 🧭 TRIP SCORE
# -------------------------------

def calculate_trip_score(data):

    matched_trip = trips[
        (trips["Departure"] == data["departure"]) &
        (trips["Arrival"] == data["arrival"])
    ]

    if matched_trip.empty:
        connection_risk = 0.5
        disruption = 0.4
        time_critical = "Medium"
    else:
        connection_risk = matched_trip["ConnectionRiskScore"].mean()
        disruption = matched_trip["DisruptionProbability"].mean()
        time_critical = matched_trip["TimeCriticality"].mode()[0]

    score = 0
    reasons = []

    # ✅ Risk gradient
    if connection_risk > 0.7:
        score += 0.4
        reasons.append("High connection risk")
    elif connection_risk > 0.5:
        score += 0.2
        reasons.append("Moderate connection risk")

    # ✅ Disruption gradient
    if disruption > 0.5:
        score += 0.3
        reasons.append("High disruption risk")
    elif disruption > 0.3:
        score += 0.15
        reasons.append("Moderate disruption")

    # ✅ Time critical
    if time_critical == "High":
        score += 0.3
        reasons.append("Time critical journey")

    # ✅ ✅ IMPORTANT: Travel duration
    trip_hours = (data["arrival_datetime"] - data["departure_datetime"]).total_seconds() / 3600

    if trip_hours <= 4:
        score += 0.1
        reasons.append("Tight travel window")
    elif trip_hours <= 6:
        score += 0.05

    if not reasons:
        reasons.append("Normal travel conditions")

    return min(score, 1), reasons


# ✅ -------------------------------
# 🧍 PASSENGER SCORE
# -------------------------------

def calculate_passenger_score(data):

    score = 0
    reasons = []

    loyalty = _normalize_loyalty_tier(data.get("loyalty_tier", ""))

    # ✅ Loyalty scoring
    if loyalty == "HON Circle":
        score += 0.4
        reasons.append("HON Circle loyalty")
    elif loyalty == "Senator":
        score += 0.3
        reasons.append("Senator loyalty")
    else:
        score += 0.2
        reasons.append("Frequent Traveller")

    # ✅ Party size
    party_size = data["party_size"]

    if party_size >= 4:
        score += 0.3
        reasons.append("Large group travel")
    elif party_size == 3:
        score += 0.25
        reasons.append("Group travel")
    elif party_size == 2:
        score += 0.2
        reasons.append("Traveling as pair")
    else:
        score += 0.1
        reasons.append("Solo traveler")

    return min(score, 1), reasons

# ✅ -------------------------------
# 💰 PRODUCT SCORING (ML Propensity ✅)
# -------------------------------

def get_ml_features(data):

    matched_trip = trips[
        (trips["Departure"] == data["departure"]) &
        (trips["Arrival"] == data["arrival"])
    ]

    if matched_trip.empty:
        risk, disruption, time = 0.5, 0.4, "Medium"
    else:
        risk = matched_trip["ConnectionRiskScore"].mean()
        disruption = matched_trip["DisruptionProbability"].mean()
        time = matched_trip["TimeCriticality"].mode()[0]

    # ✅ usage
    user = history[history["PassengerID"] == data["passenger_id"]]
    usage = user["Usage_Count"].mean() if not user.empty else 1

    # ✅ encoding
    try:
        loyalty_enc = le_loyalty.transform([data["loyalty_tier"]])[0]
    except:
        loyalty_enc = 0

    try:
        time_enc = le_time.transform([time])[0]
    except:
        time_enc = 1

    return [[risk, disruption, time_enc, loyalty_enc, usage]]


def score_products_ml(data):

    if model is None:
        return score_products(0.5, data)  # fallback

    X = get_ml_features(data)

    probs = model.predict_proba(X)[0]
    tier_probs = dict(zip(le_tier.classes_, probs))

    product_scores = []

    for _, p in products.iterrows():

        tier = str(p["ServiceTier"]).upper()
        prob = tier_probs.get(tier, 0.01)
        print(f"ML Propensity for {p['ProductName']} (Tier: {tier}): {prob}")

        # # ✅ Combine ML + SLA
        # score = prob + (p["PartnerSLAScore"] / 10)
        
        # ✅ refund
        refund_raw = p.get("RefundPolicy", "")
        if pd.isna(refund_raw) or str(refund_raw).strip() == "":
            refund_score = 0.5
        else:
            refund_policy = str(refund_raw).upper()
            if "FULL" in refund_policy or "REFUNDABLE" in refund_policy:
                refund_score = 1.0
            elif "PARTIAL" in refund_policy:
                refund_score = 0.6
            elif "NON" in refund_policy:
                refund_score = 0.3
            else:
                refund_score = 0.5
        
        sla_score = p["PartnerSLAScore"] / 5

         # # ✅ Combine ML + SLA + Refund score
        score = (
            0.6 * prob +
            0.3 * sla_score +
            0.1 * refund_score
        )
       
        product_scores.append({
            "product": p,
            "score": round(score, 4),
            "tier_affinity": prob,
            "price_affinity": 0.5,
            "sla": p["PartnerSLAScore"] / 5,
            "refund_score": refund_score
        })

    return product_scores

# ✅ -------------------------------
# 💰 PRODUCT SCORING (cold start ✅)
# -------------------------------

def score_products(base_score, data):

    product_scores = []
    min_price = products["PriceINR"].min()
    max_price = products["PriceINR"].max()

    for _, p in products.iterrows():

        # ✅ SLA score
        sla_score = p["PartnerSLAScore"] / 5

        # ✅ Price affinity
        price_pct = (p["PriceINR"] - min_price) / (max_price - min_price)
        price_affinity = 1 - abs(price_pct - 0.6)

        # ✅ ✅ NEW: Refund scoring
        refund_raw = p.get("RefundPolicy", "")

        if pd.isna(refund_raw) or str(refund_raw).strip() == "":
            refund_score = 0.5
        else:
            refund_policy = str(refund_raw).upper()

            if "FULL" in refund_policy or "REFUNDABLE" in refund_policy:
                refund_score = 1.0
            elif "PARTIAL" in refund_policy:
                refund_score = 0.6
            elif "NON" in refund_policy:
                refund_score = 0.3
            else:
                refund_score = 0.5

        # ✅ ✅ NEW: Refund importance based on risk
        refund_weight = 0.1 + (0.1 * base_score)

        # ✅ ✅ UPDATED FINAL SCORE
        score = (
            0.45 * sla_score +
            0.30 * price_affinity +
            0.15 * base_score +
            refund_weight * refund_score   # ✅ NEW
        )

        product_scores.append({
            "product": p,
            "score": round(score, 4),
            "tier_affinity": 0.5,
            "price_affinity": price_affinity,
            "sla": sla_score,
            "refund_score": refund_score  # ✅ now real value
        })

    return product_scores



# ✅ -------------------------------
# 🎯 FINAL FUNCTION
# -------------------------------
def recommend_meet_assist(data):

    data = _validate_and_normalize_input(data)

    trip_score, _ = calculate_trip_score(data)
    passenger_score, _ = calculate_passenger_score(data)

    base_score = round(
        (0.6 * trip_score + 0.4 * passenger_score),
        2
    )

    # # ✅ Step 1: base scoring
    # scored_products = score_products(base_score, data)

    
    # ✅ Detect user type (NEW ✅)
    if is_existing_user(data.get("passenger_id")):
        print("✅ Using ML Propensity Model")
        scored_products = score_products_ml(data)
    else:
        print("✅ Using Cold-start Model")
        scored_products = score_products(base_score, data)


    # ✅ ✅ NEW: GenAI refinement
    scored_products = apply_genai_adjustment(data, scored_products)

    # ✅ ranking
    ranked_products = sorted(
        scored_products,
        key=lambda x: x["score"],
        reverse=True
    )

    top_products = ranked_products[:3]
    recommendations = []

    for rank, item in enumerate(top_products, start=1):

        product = item["product"]
        recommendations.append({
            "Rank": rank,
            "Score": item["score"],
            "RecommendedProduct": product["ProductName"],
            "ServiceTier": product["ServiceTier"],
            "ServiceFeatures": product["ServiceFeatures"],
            "Price": product["PriceINR"],
            "PartnerSLAScore": product["PartnerSLAScore"],
        })

    # ✅ explanation
    explanation = generate_explanation(data, recommendations)

    return {
        "TopRecommendations": recommendations,
        "GenAI_Explanation": explanation
    }
