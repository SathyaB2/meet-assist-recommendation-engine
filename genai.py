
# from openai import OpenAI

# client = OpenAI(api_key="YOUR_API_KEY")

# recommend_meet_assist()
#         ↓
# Check: passenger_id exists?
#         ↓
#    YES ✅               NO ❌
#    ↓                   ↓
# ML Propensity       Cold-start (your current logic)
#    ↓                   ↓
#  Merge → GenAI Adjust → Ranking → Output

from dotenv import load_dotenv
import os

load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


def generate_explanation(data, recommendations):

    product_text = ""

    for r in recommendations:
        product_text += f"{r['Rank']}. {r['RecommendedProduct']} (Tier: {r['ServiceTier']}, SLA: {r['PartnerSLAScore']})\n"

    prompt = f"""
You are an AI assistant for Lufthansa Meet & Assist recommendation system.

Passenger Details:
- Loyalty Tier: {data["loyalty_tier"]}
- Party Size: {data["party_size"]}

Trip:
- Route: {data["departure"]} → {data["arrival"]}

Recommended Products:
{product_text}

Task:
Explain why these products are recommended.

Rules:
- Keep explanation very short (1-2 sentences per product)
- Focus on matching passenger needs and trip context
- Mention SLA, pricing or tier if relevant

Return:
Rank, Product, Reason
"""

    response = llm.invoke(prompt)

    return response.content
