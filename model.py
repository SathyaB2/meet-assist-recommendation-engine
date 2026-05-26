
import pandas as pd

# ✅ Load datasets
trips = pd.read_excel("trips_with_flight_1000.xlsx", engine="openpyxl")
products = pd.read_excel("meet_assist_products_full_15.xlsx", engine="openpyxl")

# ✅ Clean column names
products.columns = products.columns.str.strip()
products["PriceINR"] = pd.to_numeric(products["PriceINR"], errors="coerce")
products["PartnerSLAScore"] = pd.to_numeric(products["PartnerSLAScore"], errors="coerce")
products = products.dropna(subset=["PriceINR", "PartnerSLAScore", "ServiceTier", "ProductName", "Terminal"])


def _normalize_loyalty_tier(loyalty_tier):
    loyalty = str(loyalty_tier).strip().upper()

    if loyalty in {"HON CIRCLE", "PLATINUM"}:
        return "HON Circle"
    if loyalty in {"SENATOR", "GOLD"}:
        return "Senator"
    return "Frequent Traveller"


# ✅ -------------------------------
# 🧭 TRIP SCORE (DETERMINISTIC ✅)
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
        # ✅ FIX: stable aggregation (no random)
        connection_risk = matched_trip["ConnectionRiskScore"].mean()
        disruption = matched_trip["DisruptionProbability"].mean()
        time_critical = matched_trip["TimeCriticality"].mode()[0]

    score = 0
    reasons = []

    if connection_risk > 0.7:
        score += 0.4
        reasons.append("High connection risk")

    if disruption > 0.5:
        score += 0.3
        reasons.append("High disruption probability")

    if time_critical == "High":
        score += 0.3
        reasons.append("Time critical journey")

    return min(score, 1), reasons


# ✅ -------------------------------
# 🧍 PASSENGER SCORE
# -------------------------------
def calculate_passenger_score(data):

    score = 0
    reasons = []

    loyalty_tier = _normalize_loyalty_tier(data.get("loyalty_tier", ""))

    # ✅ Loyalty
    if loyalty_tier == "HON Circle":
        score += 0.4
        reasons.append("HON Circle loyalty")
    elif loyalty_tier == "Senator":
        score += 0.3
        reasons.append("Senator loyalty")
    else:
        score += 0.2
        reasons.append("Frequent Traveller loyalty")

    # ✅ Travel profile influence (fare removed)
    # Larger groups usually indicate higher assistance need.
    if data["party_size"] >= 4:
        score += 0.3
    elif data["party_size"] >= 2:
        score += 0.2
    else:
        score += 0.1

    # ✅ Party size
    if data["party_size"] >= 3:
        score += 0.3
        reasons.append("Group travel")
    elif data["party_size"] == 2:
        score += 0.2
    else:
        score += 0.1

    return min(score, 1), reasons


# ✅ -------------------------------
# 💰 PRODUCT + COMMERCE SCORING ✅
# -------------------------------

def _tier_preference(loyalty_tier, party_size):
    loyalty = _normalize_loyalty_tier(loyalty_tier)

    if loyalty == "HON Circle":
        ordered = ["ELITE", "PREMIUM_PLUS", "GOLD", "SILVER", "BASIC"]
    elif loyalty == "Senator":
        ordered = ["GOLD", "PREMIUM_PLUS", "ELITE", "SILVER", "BASIC"]
    else:
        ordered = ["BASIC", "SILVER", "GOLD", "PREMIUM_PLUS", "ELITE"]

    # Larger groups usually benefit from higher-touch tiers.
    if party_size >= 3:
        for tier in ["PREMIUM_PLUS", "ELITE"]:
            if tier in ordered:
                ordered.insert(0, ordered.pop(ordered.index(tier)))

    return ordered


def score_products(base_score, data):

    product_scores = []

    # ✅ Get min & max price (for normalization)
    min_price = products["PriceINR"].min()
    max_price = products["PriceINR"].max()

    party_size = int(data.get("party_size", 1) or 1)
    loyalty_tier = _normalize_loyalty_tier(data.get("loyalty_tier", ""))
    preferred_tiers = _tier_preference(loyalty_tier, party_size)

    # Budget affinity proxy without flight fare.
    if loyalty_tier == "HON Circle":
        target_price_pct = 0.75
    elif loyalty_tier == "Senator":
        target_price_pct = 0.6
    else:
        target_price_pct = 0.35

    if party_size >= 3:
        target_price_pct = min(0.9, target_price_pct + 0.1)

    for _, p in products.iterrows():

        # ✅ Normalize SLA (0–1)
        sla_score = p["PartnerSLAScore"] / 5

        # ✅ Price affinity to input budget segment.
        if max_price > min_price:
            price_pct = (p["PriceINR"] - min_price) / (max_price - min_price)
        else:
            price_pct = 0.5
        price_affinity = 1 - abs(price_pct - target_price_pct)
        price_affinity = max(0, min(price_affinity, 1))

        # ✅ Tier affinity based on loyalty + party size.
        service_tier = str(p["ServiceTier"]).upper()
        if service_tier in preferred_tiers:
            rank = preferred_tiers.index(service_tier)
            tier_affinity = max(0.2, 1 - (rank * 0.2))
        else:
            tier_affinity = 0.2

        # ✅ Group preference for higher-touch services.
        if party_size >= 3:
            group_affinity = 1.0 if service_tier in {"PREMIUM_PLUS", "ELITE", "GOLD"} else 0.6
        else:
            group_affinity = 0.8

        # ✅ FINAL SCORE (INPUT-SENSITIVE DIFFERENTIATION ✅)
        context_weight = 0.7 + (0.3 * base_score)
        score = (
            context_weight * (
                0.4 * sla_score +
                0.4 * price_affinity +
                0.2 * tier_affinity
            )
            + 0.1 * group_affinity
        )

        score = max(0, min(score, 1))

        product_scores.append((p, score, tier_affinity, price_affinity, sla_score))

    return product_scores




# ✅ -------------------------------
# 🎯 FINAL RECOMMENDATION ✅
# -------------------------------
def recommend_meet_assist(data):

    # Step 1: Scores
    trip_score, trip_reasons = calculate_trip_score(data)
    passenger_score, passenger_reasons = calculate_passenger_score(data)

    # Step 2: Base score
    base_score = round(
        (0.6 * trip_score + 0.4 * passenger_score),
        2
    )

    # Step 3: Product scoring
    scored_products = score_products(base_score, data)

    # ✅ Deterministic ranking (NO RANDOM ✅)
    ranked_products = sorted(
        scored_products,
        key=lambda x: (
            x[1],           # total score
            x[2],           # tier affinity
            x[3],           # price affinity
            x[4],           # SLA quality
            -x[0]["PriceINR"]
        ),
        reverse=True
    )

    top_products = ranked_products[:3]
    recommendations = []

    for rank, (product, raw_score, _, _, _) in enumerate(top_products, start=1):
        final_score = max(0, min(round(raw_score, 2), 1))
        recommendations.append({
            "Rank": rank,
            "FinalScore": final_score,
            "Confidence": f"{int(final_score * 100)}%",
            "RecommendedProduct": product["ProductName"],
            "ServiceTier": product["ServiceTier"],
            "Price": product["PriceINR"],
            "Terminal": product["Terminal"],
            "PartnerSLAScore": product["PartnerSLAScore"],
        })

    return {
        "Route": f"{data['departure']} → {data['arrival']}",
        "Reasons": trip_reasons + passenger_reasons,
        "TopRecommendations": recommendations,
    }
