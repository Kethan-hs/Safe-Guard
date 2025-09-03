from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
import asyncio


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Safe Guard API", description="AI-based Crime Pattern Prediction System")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global ML model and scaler (will be loaded on startup)
crime_model = None
scaler = None

# Define Models
class CrimeData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location: Dict[str, float]  # {"lat": float, "lng": float}
    crime_type: str
    severity: int  # 1-10 scale
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: str
    city: str
    description: Optional[str] = None

class CrimeDataCreate(BaseModel):
    location: Dict[str, float]
    crime_type: str
    severity: int
    state: str
    city: str
    description: Optional[str] = None

class SafetyRecommendation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location: Dict[str, float]
    risk_level: str  # "low", "medium", "high", "critical"
    risk_score: float  # 0-100
    recommendations: List[str]
    nearby_crimes: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    radius: Optional[float] = 5.0  # km

class CrimePrediction(BaseModel):
    location: Dict[str, float]
    predicted_risk_score: float
    predicted_crime_types: List[str]
    confidence: float
    factors: Dict[str, float]

# Sample Indian states and cities data
INDIAN_LOCATIONS = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Delhi": ["New Delhi", "Central Delhi", "South Delhi", "North Delhi"],
    "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Allahabad"]
}

# Crime types with severity mappings
CRIME_TYPES = {
    "theft": {"base_severity": 4, "description": "Property theft incidents"},
    "assault": {"base_severity": 6, "description": "Physical assault cases"},
    "burglary": {"base_severity": 5, "description": "Breaking and entering"},
    "robbery": {"base_severity": 7, "description": "Violent theft incidents"},
    "fraud": {"base_severity": 4, "description": "Financial fraud cases"},
    "vandalism": {"base_severity": 3, "description": "Property damage"},
    "drug_related": {"base_severity": 5, "description": "Drug-related offenses"},
    "domestic_violence": {"base_severity": 8, "description": "Domestic violence cases"},
    "cybercrime": {"base_severity": 5, "description": "Online criminal activities"},
    "traffic_violation": {"base_severity": 2, "description": "Traffic related incidents"}
}

# Initialize sample crime data
async def initialize_sample_data():
    """Initialize sample crime data for demonstration"""
    existing_count = await db.crime_data.count_documents({})
    if existing_count > 0:
        return
    
    # Sample coordinates for major Indian cities
    city_coordinates = {
        "Mumbai": {"lat": 19.0760, "lng": 72.8777},
        "Delhi": {"lat": 28.7041, "lng": 77.1025},
        "Bangalore": {"lat": 12.9716, "lng": 77.5946},
        "Chennai": {"lat": 13.0827, "lng": 80.2707},
        "Kolkata": {"lat": 22.5726, "lng": 88.3639},
        "Ahmedabad": {"lat": 23.0225, "lng": 72.5714},
        "Jaipur": {"lat": 26.9124, "lng": 75.7873},
        "Lucknow": {"lat": 26.8467, "lng": 80.9462}
    }
    
    sample_crimes = []
    for state, cities in INDIAN_LOCATIONS.items():
        for city in cities[:2]:  # Take first 2 cities per state
            if city in city_coordinates:
                base_coord = city_coordinates[city]
                for i in range(15):  # Generate 15 crimes per city
                    # Add some randomness to coordinates
                    lat_offset = np.random.uniform(-0.1, 0.1)
                    lng_offset = np.random.uniform(-0.1, 0.1)
                    
                    crime_type = np.random.choice(list(CRIME_TYPES.keys()))
                    base_severity = CRIME_TYPES[crime_type]["base_severity"]
                    severity = max(1, min(10, base_severity + np.random.randint(-2, 3)))
                    
                    # Random timestamp in last 30 days
                    days_ago = np.random.randint(0, 30)
                    hours_ago = np.random.randint(0, 24)
                    timestamp = datetime.now(timezone.utc) - pd.Timedelta(days=days_ago, hours=hours_ago)
                    
                    crime_data = CrimeData(
                        location={
                            "lat": base_coord["lat"] + lat_offset,
                            "lng": base_coord["lng"] + lng_offset
                        },
                        crime_type=crime_type,
                        severity=severity,
                        state=state,
                        city=city,
                        description=f"{crime_type.replace('_', ' ').title()} incident in {city}",
                        timestamp=timestamp
                    )
                    sample_crimes.append(crime_data.dict())
    
    if sample_crimes:
        await db.crime_data.insert_many(sample_crimes)
        logger.info(f"Initialized {len(sample_crimes)} sample crime records")

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in km"""
    from math import radians, cos, sin, asin, sqrt
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

async def get_nearby_crimes(lat: float, lng: float, radius: float = 5.0) -> List[Dict]:
    """Get crimes within radius of given coordinates"""
    crimes = await db.crime_data.find().to_list(length=None)
    nearby_crimes = []
    
    for crime in crimes:
        distance = calculate_distance(
            lat, lng,
            crime["location"]["lat"], crime["location"]["lng"]
        )
        if distance <= radius:
            crime["distance"] = round(distance, 2)
            nearby_crimes.append(crime)
    
    return sorted(nearby_crimes, key=lambda x: x["distance"])

def generate_safety_recommendations(risk_score: float, nearby_crimes: List[Dict]) -> List[str]:
    """Generate safety recommendations based on risk score and nearby crimes"""
    recommendations = []
    
    if risk_score >= 70:
        recommendations.extend([
            "⚠️ High crime area - avoid if possible, especially at night",
            "🚔 Contact local police if you notice suspicious activity",
            "👥 Travel in groups when possible"
        ])
    elif risk_score >= 40:
        recommendations.extend([
            "⚡ Stay alert and aware of your surroundings",
            "📱 Keep emergency contacts readily available"
        ])
    else:
        recommendations.extend([
            "✅ Relatively safe area, maintain normal precautions"
        ])
    
    # Crime-specific recommendations
    crime_types = [crime["crime_type"] for crime in nearby_crimes[:10]]
    
    if "theft" in crime_types or "robbery" in crime_types:
        recommendations.append("💰 Secure your valuables and avoid displaying expensive items")
    
    if "assault" in crime_types or "domestic_violence" in crime_types:
        recommendations.append("🏃‍♂️ Avoid isolated areas and trust your instincts")
    
    if "burglary" in crime_types:
        recommendations.append("🏠 Ensure your accommodation has proper security measures")
    
    if "traffic_violation" in crime_types:
        recommendations.append("🚗 Exercise extra caution while driving or crossing roads")
    
    if "cybercrime" in crime_types:
        recommendations.append("🔐 Be cautious with public WiFi and protect personal information")
    
    # Time-based recommendations
    current_hour = datetime.now().hour
    if current_hour >= 22 or current_hour <= 5:
        recommendations.append("🌙 Extra caution advised during late night hours")
    
    return recommendations[:8]  # Limit to 8 recommendations

def train_simple_crime_model():
    """Train a simple crime prediction model"""
    # This is a simplified model - in production you'd use real crime data
    # For now, we'll create a basic model based on location, time, and historical patterns
    
    # Mock training data based on crime patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Features: [hour_of_day, day_of_week, population_density, economic_factor, historical_crime_rate]
    X = np.random.rand(n_samples, 5)
    
    # Simulate realistic crime risk scores (0-100)
    y = (X[:, 0] * 30 +  # Hour of day impact
         X[:, 1] * 10 +  # Day of week impact  
         X[:, 2] * 40 +  # Population density impact
         X[:, 3] * 15 +  # Economic factor impact
         X[:, 4] * 45 +  # Historical crime rate impact
         np.random.normal(0, 5, n_samples))  # Add noise
    
    y = np.clip(y, 0, 100)  # Ensure scores are between 0-100
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    return model, scaler

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Safe Guard API - Crime Pattern Prediction System"}

@api_router.get("/locations")
async def get_indian_locations():
    """Get list of Indian states and cities"""
    return {"locations": INDIAN_LOCATIONS}

@api_router.get("/crime-types")
async def get_crime_types():
    """Get available crime types"""
    return {"crime_types": CRIME_TYPES}

@api_router.post("/crime-data", response_model=CrimeData)
async def create_crime_data(input: CrimeDataCreate):
    """Add new crime data"""
    crime_dict = input.dict()
    crime_obj = CrimeData(**crime_dict)
    await db.crime_data.insert_one(crime_obj.dict())
    return crime_obj

@api_router.get("/crime-data", response_model=List[CrimeData])
async def get_crime_data(state: Optional[str] = None, city: Optional[str] = None, limit: int = 100):
    """Get crime data with optional filtering"""
    query = {}
    if state:
        query["state"] = state
    if city:
        query["city"] = city
    
    crimes = await db.crime_data.find(query).limit(limit).to_list(length=None)
    return [CrimeData(**crime) for crime in crimes]

@api_router.post("/safety-analysis", response_model=SafetyRecommendation)
async def analyze_location_safety(location_request: LocationRequest):
    """Analyze safety of a specific location"""
    lat, lng, radius = location_request.latitude, location_request.longitude, location_request.radius
    
    # Get nearby crimes
    nearby_crimes = await get_nearby_crimes(lat, lng, radius)
    
    # Calculate risk score based on nearby crimes
    if not nearby_crimes:
        risk_score = 10  # Very low risk if no nearby crimes
        risk_level = "low"
    else:
        # Weight crimes by severity and recency
        total_risk = 0
        for crime in nearby_crimes:
            severity_weight = crime["severity"] / 10.0
            distance_weight = max(0.1, 1 - (crime["distance"] / radius))
            
            # Time decay (more recent crimes have higher weight)
            crime_date = crime["timestamp"]
            if isinstance(crime_date, str):
                crime_date = datetime.fromisoformat(crime_date.replace('Z', '+00:00'))
            days_ago = (datetime.now(timezone.utc) - crime_date).days
            time_weight = max(0.1, 1 - (days_ago / 30))
            
            crime_risk = severity_weight * distance_weight * time_weight * 10
            total_risk += crime_risk
        
        risk_score = min(100, total_risk)
        
        if risk_score >= 70:
            risk_level = "critical"
        elif risk_score >= 50:
            risk_level = "high"
        elif risk_score >= 25:
            risk_level = "medium"
        else:
            risk_level = "low"
    
    # Generate recommendations
    recommendations = generate_safety_recommendations(risk_score, nearby_crimes)
    
    # Prepare nearby crimes data (limit to 10 most relevant)
    crimes_data = []
    for crime in nearby_crimes[:10]:
        crimes_data.append({
            "crime_type": crime["crime_type"],
            "severity": crime["severity"],
            "distance": crime["distance"],
            "timestamp": crime["timestamp"],
            "description": crime.get("description", "")
        })
    
    safety_recommendation = SafetyRecommendation(
        location={"lat": lat, "lng": lng},
        risk_level=risk_level,
        risk_score=round(risk_score, 2),
        recommendations=recommendations,
        nearby_crimes=crimes_data
    )
    
    return safety_recommendation

@api_router.post("/predict-crime", response_model=CrimePrediction)
async def predict_crime_risk(location_request: LocationRequest):
    """Predict crime risk using ML model"""
    if crime_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML model not initialized")
    
    lat, lng = location_request.latitude, location_request.longitude
    
    # Extract features for prediction
    current_time = datetime.now()
    hour_of_day = current_time.hour / 24.0
    day_of_week = current_time.weekday() / 7.0
    
    # Mock features (in production, these would come from real data sources)
    population_density = 0.6  # Mock population density
    economic_factor = 0.5     # Mock economic indicator
    
    # Get historical crime rate from nearby crimes
    nearby_crimes = await get_nearby_crimes(lat, lng, 10.0)  # 10km radius
    historical_crime_rate = min(1.0, len(nearby_crimes) / 50.0)
    
    features = np.array([[hour_of_day, day_of_week, population_density, economic_factor, historical_crime_rate]])
    features_scaled = scaler.transform(features)
    
    predicted_risk = crime_model.predict(features_scaled)[0]
    confidence = 0.75  # Mock confidence score
    
    # Predict likely crime types based on nearby patterns
    crime_type_counts = {}
    for crime in nearby_crimes[:20]:
        crime_type = crime["crime_type"]
        crime_type_counts[crime_type] = crime_type_counts.get(crime_type, 0) + 1
    
    predicted_crime_types = sorted(crime_type_counts.keys(), 
                                 key=lambda x: crime_type_counts[x], 
                                 reverse=True)[:3]
    
    if not predicted_crime_types:
        predicted_crime_types = ["theft", "traffic_violation"]
    
    factors = {
        "time_of_day": hour_of_day,
        "population_density": population_density,
        "historical_crimes": historical_crime_rate,
        "economic_factor": economic_factor
    }
    
    return CrimePrediction(
        location={"lat": lat, "lng": lng},
        predicted_risk_score=round(predicted_risk, 2),
        predicted_crime_types=predicted_crime_types,
        confidence=confidence,
        factors=factors
    )

@api_router.get("/crime-heatmap")
async def get_crime_heatmap(state: Optional[str] = None, days: int = 30):
    """Get crime heatmap data for visualization"""
    query = {}
    if state:
        query["state"] = state
    
    # Filter by date range
    cutoff_date = datetime.now(timezone.utc) - pd.Timedelta(days=days)
    query["timestamp"] = {"$gte": cutoff_date}
    
    crimes = await db.crime_data.find(query).to_list(length=None)
    
    heatmap_data = []
    for crime in crimes:
        heatmap_data.append({
            "lat": crime["location"]["lat"],
            "lng": crime["location"]["lng"],
            "intensity": crime["severity"] / 10.0,
            "crime_type": crime["crime_type"],
            "timestamp": crime["timestamp"]
        })
    
    return {"heatmap_data": heatmap_data}

@api_router.get("/dashboard-stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    total_crimes = await db.crime_data.count_documents({})
    
    # Get crimes by type
    pipeline = [
        {"$group": {"_id": "$crime_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    crime_by_type = await db.crime_data.aggregate(pipeline).to_list(length=None)
    
    # Get crimes by state
    pipeline = [
        {"$group": {"_id": "$state", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    crime_by_state = await db.crime_data.aggregate(pipeline).to_list(length=None)
    
    # Recent activity
    recent_crimes = await db.crime_data.find().sort("timestamp", -1).limit(10).to_list(length=None)
    
    return {
        "total_crimes": total_crimes,
        "crime_by_type": crime_by_type,
        "crime_by_state": crime_by_state,
        "recent_crimes": recent_crimes
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global crime_model, scaler
    logger.info("Starting Safe Guard API...")
    
    # Initialize sample data
    await initialize_sample_data()
    
    # Train ML model
    crime_model, scaler = train_simple_crime_model()
    logger.info("ML model initialized successfully")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()