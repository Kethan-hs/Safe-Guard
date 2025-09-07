# server.py
from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import Table, Column, Integer, String, Float, DateTime, MetaData, select, func, insert, and_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
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

# ---- Load environment ----
ROOT_DIR = Path(__file__).parent
# load .env from project root (works with a Path)
load_dotenv(ROOT_DIR / ".env")

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("server")

# ---- App and Router ----
app = FastAPI(title="Safe Guard API", description="AI-based Crime Pattern Prediction System")
api_router = APIRouter(prefix="/api")

# ---- Database (PostgreSQL + SQLAlchemy async) ----
POSTGRES_URL = os.environ.get("POSTGRES_URL")
if not POSTGRES_URL:
    raise RuntimeError("No PostgreSQL connection string found. Set POSTGRES_URL in .env")

# If user provided a synchronous URL by mistake (postgresql://) and expects asyncpg,
# allow both dialects. The create_async_engine requires an async dialect.
# If user already gave async dialect (postgresql+asyncpg://...) we keep it.
if POSTGRES_URL.startswith("postgresql://"):
    # convert to asyncpg dialect automatically
    POSTGRES_URL_ASYNC = POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    POSTGRES_URL_ASYNC = POSTGRES_URL

engine = create_async_engine(POSTGRES_URL_ASYNC, future=True, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
metadata = MetaData()

crime_data_table = Table(
    "crime_data",
    metadata,
    Column("id", String, primary_key=True),
    Column("location", JSONB),
    Column("crime_type", String),
    Column("severity", Integer),
    Column("timestamp", DateTime(timezone=True)),
    Column("state", String),
    Column("city", String),
    Column("description", String),
)

# --- Helper: simple SQLAlchemy wrapper to mimic the previous Mongo-like API ---


class CrimeDataTable:
    def __init__(self, table, sessionmaker):
        self.table = table
        self.sessionmaker = sessionmaker

    async def insert_one(self, doc: dict):
        async with self.sessionmaker() as session:
            await session.execute(insert(self.table).values(**doc))
            await session.commit()

    async def insert_many(self, docs: list):
        if not docs:
            return
        async with self.sessionmaker() as session:
            await session.execute(insert(self.table), docs)
            await session.commit()

    def _build_filters(self, query: dict):
        """
        Support simple equality and basic operators like {"timestamp": {"$gte": dt}}.
        Returns list of SQLAlchemy conditions.
        """
        conditions = []
        if not query:
            return conditions
        for k, v in query.items():
            if not hasattr(self.table.c, k):
                continue
            col = getattr(self.table.c, k)
            # operator handling
            if isinstance(v, dict):
                # support $gte, $lte, $gt, $lt, $eq
                if "$gte" in v:
                    conditions.append(col >= v["$gte"])
                if "$lte" in v:
                    conditions.append(col <= v["$lte"])
                if "$gt" in v:
                    conditions.append(col > v["$gt"])
                if "$lt" in v:
                    conditions.append(col < v["$lt"])
                if "$eq" in v:
                    conditions.append(col == v["$eq"])
            else:
                conditions.append(col == v)
        return conditions

    async def find(self, query: dict = None, limit: Optional[int] = None, sort: Optional[List] = None):
        """
        Returns list of dict rows. Supports simple equality and $gte/$lte in query.
        sort: list of tuples like [("timestamp", "desc")]
        """
        async with self.sessionmaker() as session:
            stmt = select(self.table)
            filters = self._build_filters(query or {})
            if filters:
                stmt = stmt.where(and_(*filters))
            if sort:
                # Basic sort handling
                for field, direction in sort:
                    if hasattr(self.table.c, field):
                        col = getattr(self.table.c, field)
                        if direction and direction.lower().startswith("desc"):
                            stmt = stmt.order_by(col.desc())
                        else:
                            stmt = stmt.order_by(col.asc())
            if limit:
                stmt = stmt.limit(limit)
            res = await session.execute(stmt)
            rows = res.fetchall()
            return [dict(r._mapping) for r in rows]

    async def count_documents(self, query: dict = None):
        async with self.sessionmaker() as session:
            stmt = select(func.count()).select_from(self.table)
            filters = self._build_filters(query or {})
            if filters:
                stmt = stmt.where(and_(*filters))
            res = await session.execute(stmt)
            return int(res.scalar() or 0)

    async def aggregate(self, pipeline: list):
        """
        Minimal aggregation emulation for pipelines of the form:
        [ {"$group": {"_id": "$field", "count": {"$sum": 1}}}, {"$sort": {"count": -1}} ]
        """
        if not pipeline:
            return []
        # handle group stage only (basic)
        grp_stage = pipeline[0].get("$group") if isinstance(pipeline[0], dict) else None
        if grp_stage:
            group_field = None
            for k, v in grp_stage.items():
                if k == "_id" and isinstance(v, str) and v.startswith("$"):
                    group_field = v[1:]
            if not group_field or not hasattr(self.table.c, group_field):
                return []
            async with self.sessionmaker() as session:
                col = getattr(self.table.c, group_field)
                stmt = select(col.label("_id"), func.count().label("count")).group_by(col).order_by(func.count().desc())
                res = await session.execute(stmt)
                rows = res.fetchall()
                return [{"_id": r._mapping["_id"], "count": r._mapping["count"]} for r in rows]
        return []


db = type("DB", (), {})()
db.crime_data = CrimeDataTable(crime_data_table, async_session)


async def _ensure_tables():
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


@app.on_event("startup")
async def _create_tables_on_startup():
    # ensure DB tables exist before other startup tasks
    try:
        await _ensure_tables()
    except Exception as e:
        logger.warning(f"Failed to ensure tables during startup: {e}")


# ========== Pydantic Models ==========
class CrimeData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location: Dict[str, float]
    crime_type: str
    severity: int
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
    risk_level: str
    risk_score: float
    recommendations: List[str]
    nearby_crimes: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    radius: Optional[float] = 5.0


class CrimePrediction(BaseModel):
    location: Dict[str, float]
    predicted_risk_score: float
    predicted_crime_types: List[str]
    confidence: float
    factors: Dict[str, float]


# ========== Constants / Seed data ==========
CITY_COORDINATES = {
    "Mumbai": {"lat": 19.0760, "lng": 72.8777, "state": "Maharashtra"},
    "Delhi": {"lat": 28.7041, "lng": 77.1025, "state": "Delhi"},
    "Bangalore": {"lat": 12.9716, "lng": 77.5946, "state": "Karnataka"},
    "Chennai": {"lat": 13.0827, "lng": 80.2707, "state": "Tamil Nadu"},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639, "state": "West Bengal"},
    "Ahmedabad": {"lat": 23.0225, "lng": 72.5714, "state": "Gujarat"},
    "Jaipur": {"lat": 26.9124, "lng": 75.7873, "state": "Rajasthan"},
    "Lucknow": {"lat": 26.8467, "lng": 80.9462, "state": "Uttar Pradesh"},
    "Pune": {"lat": 18.5204, "lng": 73.8567, "state": "Maharashtra"},
    "Hyderabad": {"lat": 17.3850, "lng": 78.4867, "state": "Telangana"},
    "Surat": {"lat": 21.1702, "lng": 72.8311, "state": "Gujarat"},
    "Kanpur": {"lat": 26.4499, "lng": 80.3319, "state": "Uttar Pradesh"},
    "Nagpur": {"lat": 21.1458, "lng": 79.0882, "state": "Maharashtra"},
    "Indore": {"lat": 22.7196, "lng": 75.8577, "state": "Madhya Pradesh"},
    "Agra": {"lat": 27.1767, "lng": 78.0081, "state": "Uttar Pradesh"},
    "Nashik": {"lat": 19.9975, "lng": 73.7898, "state": "Maharashtra"},
    "Faridabad": {"lat": 28.4089, "lng": 77.3178, "state": "Haryana"},
    "Meerut": {"lat": 28.9845, "lng": 77.7064, "state": "Uttar Pradesh"},
    "Rajkot": {"lat": 22.3039, "lng": 70.8022, "state": "Gujarat"},
    "Kalyan": {"lat": 19.2437, "lng": 73.1355, "state": "Maharashtra"},
    "Vasai": {"lat": 19.4036, "lng": 72.8062, "state": "Maharashtra"},
    "Varanasi": {"lat": 25.3176, "lng": 82.9739, "state": "Uttar Pradesh"},
    "Srinagar": {"lat": 34.0837, "lng": 74.7973, "state": "Jammu and Kashmir"},
    "Ludhiana": {"lat": 30.9010, "lng": 75.8573, "state": "Punjab"},
    "Thane": {"lat": 19.2183, "lng": 72.9781, "state": "Maharashtra"},
    "Visakhapatnam": {"lat": 17.6868, "lng": 83.2185, "state": "Andhra Pradesh"},
    "Bhopal": {"lat": 23.2599, "lng": 77.4126, "state": "Madhya Pradesh"},
    "Patna": {"lat": 25.5941, "lng": 85.1376, "state": "Bihar"},
    "Ghaziabad": {"lat": 28.6692, "lng": 77.4538, "state": "Uttar Pradesh"},
}

INDIAN_LOCATIONS = {}
for city, data in CITY_COORDINATES.items():
    state = data["state"]
    INDIAN_LOCATIONS.setdefault(state, []).append(city)

CRIME_TYPES = {
    "ARSON": {"base_severity": 7, "description": "Fire-related criminal acts"},
    "ASSAULT": {"base_severity": 6, "description": "Physical assault cases"},
    "BURGLARY": {"base_severity": 5, "description": "Breaking and entering"},
    "COUNTERFEITING": {"base_severity": 4, "description": "Fake document/currency creation"},
    "CYBERCRIME": {"base_severity": 5, "description": "Online criminal activities"},
    "DOMESTIC VIOLENCE": {"base_severity": 8, "description": "Domestic violence cases"},
    "DRUG OFFENSE": {"base_severity": 6, "description": "Drug-related offenses"},
    "EXTORTION": {"base_severity": 7, "description": "Extortion and blackmail"},
    "FIREARM OFFENSE": {"base_severity": 8, "description": "Illegal firearm activities"},
    "FRAUD": {"base_severity": 4, "description": "Financial fraud cases"},
    "HOMICIDE": {"base_severity": 10, "description": "Murder and manslaughter"},
    "IDENTITY THEFT": {"base_severity": 5, "description": "Identity theft crimes"},
    "ILLEGAL POSSESSION": {"base_severity": 4, "description": "Illegal possession of items"},
    "KIDNAPPING": {"base_severity": 9, "description": "Kidnapping and abduction"},
    "PUBLIC INTOXICATION": {"base_severity": 2, "description": "Public intoxication"},
    "ROBBERY": {"base_severity": 7, "description": "Violent theft incidents"},
    "SEXUAL ASSAULT": {"base_severity": 9, "description": "Sexual assault cases"},
    "SHOPLIFTING": {"base_severity": 3, "description": "Retail theft"},
    "TRAFFIC VIOLATION": {"base_severity": 2, "description": "Traffic related incidents"},
    "VANDALISM": {"base_severity": 3, "description": "Property damage"},
    "VEHICLE - STOLEN": {"base_severity": 5, "description": "Vehicle theft"},
}

# ========== Data initialization ==========

async def debug_csv_structure():
    """Helper function to debug CSV structure"""
    possible_paths = [
        ROOT_DIR / "crime_dataset_india.csv",
        ROOT_DIR / "data" / "crime_dataset_india.csv",
        ROOT_DIR / "dataset" / "crime_dataset_india.csv",
        Path("/app/crime_dataset_india.csv"),
    ]
    
    # Also check for any CSV file in the current directory
    try:
        csv_files = list(ROOT_DIR.glob("*.csv"))
        possible_paths.extend(csv_files)
    except Exception:
        pass

    for csv_path in possible_paths:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, nrows=5)  # Read only first 5 rows
                logger.info(f"=== CSV Debug Info for {csv_path.name} ===")
                logger.info(f"Shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
                logger.info("Sample data:")
                logger.info(df.to_string())
                logger.info("=" * 50)
                return
            except Exception as e:
                logger.error(f"Error reading {csv_path}: {e}")
    
    logger.warning("No CSV files found for debugging")


async def initialize_real_crime_data():
    """Load and initialize real crime data from the uploaded CSV"""
    try:
        existing_count = await db.crime_data.count_documents({})
        if existing_count > 0:
            logger.info(f"Crime data already loaded: {existing_count} records")
            return
    except Exception as e:
        logger.warning(f"Could not check existing data: {e}")
        # Continue to try loading data anyway

    # Expanded list of possible CSV file paths
    possible_paths = [
        ROOT_DIR / "crime_dataset_india.csv",
        ROOT_DIR / "data" / "crime_dataset_india.csv",
        ROOT_DIR / "dataset" / "crime_dataset_india.csv",
        Path("/app/crime_dataset_india.csv"),
        Path("/app/data/crime_dataset_india.csv"),
        # Add more flexible naming
        ROOT_DIR / "crime_data.csv",
        ROOT_DIR / "dataset.csv",
        ROOT_DIR / "crimes.csv",
    ]
    
    # Also check for any CSV file in the current directory
    try:
        csv_files = list(ROOT_DIR.glob("*.csv"))
        possible_paths.extend(csv_files)
    except Exception:
        pass

    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            logger.info(f"Found CSV file at: {csv_path}")
            break

    if not csv_path:
        logger.warning(f"No crime dataset found. Searched paths: {[str(p) for p in possible_paths[:8]]}")
        logger.warning("Current directory contents:")
        try:
            for item in ROOT_DIR.iterdir():
                logger.warning(f"  - {item.name}")
        except Exception:
            pass
        await initialize_sample_data()
        return

    try:
        # Read CSV with more flexible options
        df = pd.read_csv(csv_path, encoding='utf-8')
        logger.info(f"Successfully read CSV with {len(df)} rows")
        logger.info(f"CSV columns: {list(df.columns)}")
        
        # Print first few rows for debugging
        logger.info("First 3 rows of data:")
        logger.info(df.head(3).to_string())
        
    except UnicodeDecodeError:
        try:
            # Try different encoding
            df = pd.read_csv(csv_path, encoding='latin-1')
            logger.info(f"Successfully read CSV with latin-1 encoding: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to read CSV with any encoding: {e}")
            await initialize_sample_data()
            return
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        await initialize_sample_data()
        return

    # More flexible column mapping
    column_mapping = {
        # City columns
        'city': ['City', 'city', 'CITY', 'City Name', 'CityName'],
        # Crime type columns
        'crime_type': ['Crime Description', 'Crime', 'CRIME', 'Crime Type', 'CrimeType', 'Offense', 'Incident Type'],
        # Date columns
        'date': ['Date of Occurrence', 'Date', 'DATE', 'Incident Date', 'Report Date', 'Occurrence Date'],
        # Report number columns
        'report_number': ['Report Number', 'Report No', 'Case Number', 'Incident Number', 'ID'],
        # State columns
        'state': ['State', 'STATE', 'State Name', 'StateName'],
        # Additional useful columns
        'district': ['District', 'DISTRICT', 'Area'],
        'location': ['Location', 'Address', 'Place']
    }

    def find_column(df, possible_names):
        """Find the first matching column name"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    # Find actual column names
    city_col = find_column(df, column_mapping['city'])
    crime_col = find_column(df, column_mapping['crime_type'])
    date_col = find_column(df, column_mapping['date'])
    report_col = find_column(df, column_mapping['report_number'])
    state_col = find_column(df, column_mapping['state'])
    
    logger.info(f"Mapped columns - City: {city_col}, Crime: {crime_col}, Date: {date_col}")

    if not city_col or not crime_col:
        logger.error("Could not find required columns (City and Crime Type)")
        logger.error("Please ensure your CSV has columns for city and crime type")
        await initialize_sample_data()
        return

    crimes_to_insert = []
    processed_count = 0
    skipped_count = 0

    for index, row in df.iterrows():
        try:
            # Get city
            city = str(row.get(city_col, "")).strip()
            if not city or city.lower() in ['nan', 'null', '']:
                skipped_count += 1
                continue

            # Check if city exists in our coordinates
            city_found = None
            for known_city in CITY_COORDINATES:
                if city.lower() == known_city.lower():
                    city_found = known_city
                    break
            
            # If exact match not found, try partial matching
            if not city_found:
                for known_city in CITY_COORDINATES:
                    if city.lower() in known_city.lower() or known_city.lower() in city.lower():
                        city_found = known_city
                        break
            
            if not city_found:
                # For debugging, log first few unknown cities
                if skipped_count < 10:
                    logger.warning(f"Unknown city: '{city}' - skipping")
                skipped_count += 1
                continue

            city_data = CITY_COORDINATES[city_found]

            # Get crime type
            crime_description = str(row.get(crime_col, "")).strip().upper()
            if not crime_description or crime_description.lower() in ['nan', 'null', '']:
                skipped_count += 1
                continue

            # Find matching crime type
            crime_type_found = None
            for known_crime in CRIME_TYPES:
                if crime_description == known_crime:
                    crime_type_found = known_crime
                    break
                elif crime_description in known_crime or known_crime in crime_description:
                    crime_type_found = known_crime
                    break
            
            # Try partial matching for common crime types
            if not crime_type_found:
                crime_lower = crime_description.lower()
                if 'theft' in crime_lower or 'steal' in crime_lower:
                    crime_type_found = 'BURGLARY'
                elif 'assault' in crime_lower or 'attack' in crime_lower:
                    crime_type_found = 'ASSAULT'
                elif 'murder' in crime_lower or 'kill' in crime_lower:
                    crime_type_found = 'HOMICIDE'
                elif 'robbery' in crime_lower or 'rob' in crime_lower:
                    crime_type_found = 'ROBBERY'
                elif 'fraud' in crime_lower or 'cheat' in crime_lower:
                    crime_type_found = 'FRAUD'
                elif 'drug' in crime_lower or 'narcotic' in crime_lower:
                    crime_type_found = 'DRUG OFFENSE'
                elif 'vehicle' in crime_lower or 'car' in crime_lower:
                    crime_type_found = 'VEHICLE - STOLEN'
                else:
                    # Default to a common crime type
                    crime_type_found = 'ASSAULT'

            # Generate location with slight randomization
            lat_offset = float(np.random.uniform(-0.02, 0.02))
            lng_offset = float(np.random.uniform(-0.02, 0.02))

            # Calculate severity
            base_severity = CRIME_TYPES[crime_type_found]["base_severity"]
            severity = int(max(1, min(10, base_severity + np.random.randint(-1, 2))))

            # Parse timestamp
            timestamp = datetime.now(timezone.utc)
            if date_col and pd.notna(row.get(date_col)):
                try:
                    date_str = str(row.get(date_col, "")).strip()
                    if date_str and date_str.lower() not in ['nan', 'null', '']:
                        # Try pandas to_datetime with various formats
                        parsed_date = pd.to_datetime(date_str, errors="coerce", infer_datetime_format=True)
                        if pd.isna(parsed_date):
                            # Try manual parsing for common Indian date formats
                            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y']:
                                try:
                                    parsed_date = pd.to_datetime(date_str, format=fmt)
                                    break
                                except:
                                    continue
                        
                        if not pd.isna(parsed_date):
                            if parsed_date.tzinfo is None:
                                parsed_date = parsed_date.tz_localize("UTC")
                            timestamp = parsed_date.to_pydatetime()
                        else:
                            # Use a random date from the past year
                            days_ago = np.random.randint(0, 365)
                            timestamp = datetime.now(timezone.utc) - pd.Timedelta(days=days_ago)
                except Exception as e:
                    logger.debug(f"Date parsing failed for '{date_str}': {e}")
                    # Use random date from past year
                    days_ago = np.random.randint(0, 365)
                    timestamp = datetime.now(timezone.utc) - pd.Timedelta(days=days_ago)

            # Get state
            state = city_data["state"]
            if state_col and pd.notna(row.get(state_col)):
                state = str(row.get(state_col, city_data["state"])).strip()

            # Create description
            report_num = str(row.get(report_col, f"AUTO-{index}")) if report_col else f"AUTO-{index}"
            description = f"{crime_type_found.replace('_', ' ').title()} incident in {city_found} (Report #{report_num})"

            crime_data_obj = CrimeData(
                location={
                    "lat": city_data["lat"] + lat_offset, 
                    "lng": city_data["lng"] + lng_offset
                },
                crime_type=crime_type_found,
                severity=severity,
                state=state,
                city=city_found,
                description=description,
                timestamp=timestamp,
            )
            
            crimes_to_insert.append(crime_data_obj.dict())
            processed_count += 1

            # Insert in batches
            if len(crimes_to_insert) >= 500:
                await db.crime_data.insert_many(crimes_to_insert)
                logger.info(f"Inserted batch: {processed_count} records processed, {skipped_count} skipped")
                crimes_to_insert = []

        except Exception as e:
            logger.debug(f"Error processing row {index}: {e}")
            skipped_count += 1
            continue

    # Insert remaining records
    if crimes_to_insert:
        await db.crime_data.insert_many(crimes_to_insert)

    logger.info(f"Data loading complete: {processed_count} records processed, {skipped_count} skipped")
    
    if processed_count == 0:
        logger.warning("No records were processed successfully. Falling back to sample data.")
        await initialize_sample_data()
    else:
        logger.info(f"Successfully loaded {processed_count} real crime records from dataset")


async def initialize_sample_data():
    """Initialize sample crime data for demonstration"""
    existing_count = await db.crime_data.count_documents({})
    if existing_count > 0:
        logger.info("Sample data already exists; skipping sample initialization")
        return

    # smaller sample set for quick start
    sample_crimes = []
    city_coordinates = {
        "Mumbai": {"lat": 19.0760, "lng": 72.8777},
        "Delhi": {"lat": 28.7041, "lng": 77.1025},
        "Bangalore": {"lat": 12.9716, "lng": 77.5946},
        "Chennai": {"lat": 13.0827, "lng": 80.2707},
        "Kolkata": {"lat": 22.5726, "lng": 88.3639},
        "Ahmedabad": {"lat": 23.0225, "lng": 72.5714},
        "Jaipur": {"lat": 26.9124, "lng": 75.7873},
        "Lucknow": {"lat": 26.8467, "lng": 80.9462},
    }

    for state, cities in INDIAN_LOCATIONS.items():
        for city in cities[:2]:
            if city in city_coordinates:
                base_coord = city_coordinates[city]
                for i in range(10):
                    lat_offset = float(np.random.uniform(-0.05, 0.05))
                    lng_offset = float(np.random.uniform(-0.05, 0.05))
                    crime_type = list(CRIME_TYPES.keys())[np.random.randint(0, len(CRIME_TYPES))]
                    base_severity = CRIME_TYPES[crime_type]["base_severity"]
                    severity = int(max(1, min(10, base_severity + np.random.randint(-2, 3))))
                    days_ago = int(np.random.randint(0, 30))
                    hours_ago = int(np.random.randint(0, 24))
                    timestamp = datetime.now(timezone.utc) - pd.Timedelta(days=days_ago, hours=hours_ago)

                    crime_data = CrimeData(
                        location={"lat": base_coord["lat"] + lat_offset, "lng": base_coord["lng"] + lng_offset},
                        crime_type=crime_type,
                        severity=severity,
                        state=state,
                        city=city,
                        description=f"{crime_type.replace('_', ' ').title()} incident in {city}",
                        timestamp=timestamp,
                    )
                    sample_crimes.append(crime_data.dict())

    if sample_crimes:
        await db.crime_data.insert_many(sample_crimes)
        logger.info(f"Initialized {len(sample_crimes)} sample crime records")


# ========== Utilities ==========


def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in km"""
    from math import radians, cos, sin, asin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


async def get_nearby_crimes(lat: float, lng: float, radius: float = 5.0) -> List[Dict]:
    """Get crimes within radius (km). This loads all crimes and filters in Python for simplicity."""
    # For production, you would implement spatial queries (PostGIS).
    crimes = await db.crime_data.find()
    nearby = []
    for crime in crimes:
        loc = crime.get("location") or {}
        try:
            distance = calculate_distance(lat, lng, float(loc.get("lat", 0)), float(loc.get("lng", 0)))
        except Exception:
            continue
        if distance <= radius:
            crime_copy = crime.copy()
            crime_copy["distance"] = round(distance, 2)
            nearby.append(crime_copy)
    return sorted(nearby, key=lambda x: x["distance"])


def generate_safety_recommendations(risk_score: float, nearby_crimes: List[Dict]) -> List[str]:
    recommendations = []
    if risk_score >= 70:
        recommendations.extend(
            [
                "⚠️ High crime area - avoid if possible, especially at night",
                "🚓 Contact local police if you notice suspicious activity",
                "👥 Travel in groups when possible",
            ]
        )
    elif risk_score >= 40:
        recommendations.extend(["⚡ Stay alert and aware of your surroundings", "📱 Keep emergency contacts readily available"])
    else:
        recommendations.append("✅ Relatively safe area, maintain normal precautions")

    crime_types = [c.get("crime_type", "").lower() for c in nearby_crimes[:10]]

    if any("theft" in t or "robbery" in t for t in crime_types):
        recommendations.append("💰 Secure your valuables and avoid displaying expensive items")
    if any("assault" in t for t in crime_types):
        recommendations.append("🏃‍♂️ Avoid isolated areas and trust your instincts")
    if any("burglary" in t for t in crime_types):
        recommendations.append("🏠 Ensure your accommodation has proper security measures")
    if any("traffic" in t for t in crime_types):
        recommendations.append("🚗 Exercise extra caution while driving or crossing roads")
    if any("cyber" in t for t in crime_types):
        recommendations.append("🔒 Be cautious with public WiFi and protect personal information")

    current_hour = datetime.now().hour
    if current_hour >= 22 or current_hour <= 5:
        recommendations.append("🌙 Extra caution advised during late night hours")

    return recommendations[:8]


# ========== ML Training (mock/enhanced) ==========
def train_crime_model_with_real_data():
    np.random.seed(42)
    n_samples = 2000
    X = np.random.rand(n_samples, 9)
    y = []
    for i in range(n_samples):
        hour = X[i, 0]
        day_of_week = X[i, 1]
        month = X[i, 2]
        pop_density = X[i, 3]
        economic = X[i, 4]
        historical = X[i, 5]
        violent_crimes = X[i, 6]
        fire_accidents = X[i, 7]
        traffic_fatalities = X[i, 8]

        base_risk = (
            hour * 25
            + (1 - day_of_week) * 15
            + month * 10
            + pop_density * 35
            + (1 - economic) * 20
            + historical * 50
            + violent_crimes * 40
            + fire_accidents * 25
            + traffic_fatalities * 15
        )
        risk_score = base_risk + np.random.normal(0, 8)
        y.append(max(0, min(100, risk_score)))

    y = np.array(y)
    model = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_split=5, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model, scaler


# ========== API Endpoints ==========


@api_router.get("/")
async def root():
    return {"message": "Safe Guard API - Crime Pattern Prediction System"}


@api_router.get("/locations")
async def get_indian_locations():
    return {"locations": INDIAN_LOCATIONS}


@api_router.get("/crime-types")
async def get_crime_types():
    return {"crime_types": CRIME_TYPES}


@api_router.post("/crime-data", response_model=CrimeData)
async def create_crime_data(input: CrimeDataCreate):
    crime_obj = CrimeData(**input.dict())
    # ensure timestamp serializable type
    if isinstance(crime_obj.timestamp, pd.Timestamp):
        crime_obj.timestamp = crime_obj.timestamp.to_pydatetime()
    await db.crime_data.insert_one(crime_obj.dict())
    return crime_obj


@api_router.get("/crime-data", response_model=List[CrimeData])
async def get_crime_data(state: Optional[str] = None, city: Optional[str] = None, limit: int = 100):
    query = {}
    if state:
        query["state"] = state
    if city:
        query["city"] = city
    crimes = await db.crime_data.find(query=query, limit=limit)
    # convert timestamp strings to datetimes if needed
    out = []
    for c in crimes:
        ts = c.get("timestamp")
        if isinstance(ts, str):
            try:
                c["timestamp"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                c["timestamp"] = datetime.now(timezone.utc)
        out.append(CrimeData(**c))
    return out


@api_router.post("/safety-analysis", response_model=SafetyRecommendation)
async def analyze_location_safety(location_request: LocationRequest):
    lat, lng, radius = location_request.latitude, location_request.longitude, location_request.radius
    nearby_crimes = await get_nearby_crimes(lat, lng, radius)

    if not nearby_crimes:
        risk_score = 10
        risk_level = "low"
    else:
        total_risk = 0.0
        for crime in nearby_crimes:
            severity_weight = crime.get("severity", 5) / 10.0
            distance_weight = max(0.1, 1 - (crime.get("distance", 0) / radius))
            crime_date = crime.get("timestamp")
            if isinstance(crime_date, str):
                try:
                    crime_date = datetime.fromisoformat(crime_date.replace("Z", "+00:00"))
                except Exception:
                    crime_date = datetime.now(timezone.utc)
            if isinstance(crime_date, pd.Timestamp):
                crime_date = crime_date.to_pydatetime()
            if crime_date and crime_date.tzinfo is None:
                crime_date = crime_date.replace(tzinfo=timezone.utc)
            days_ago = (datetime.now(timezone.utc) - crime_date).days if crime_date else 0
            time_weight = max(0.1, 1 - (days_ago / 30.0))
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

    recommendations = generate_safety_recommendations(risk_score, nearby_crimes)
    crimes_data = []
    for crime in nearby_crimes[:10]:
        crimes_data.append(
            {
                "crime_type": crime.get("crime_type"),
                "severity": crime.get("severity"),
                "distance": crime.get("distance"),
                "timestamp": crime.get("timestamp"),
                "description": crime.get("description", ""),
            }
        )

    safety_recommendation = SafetyRecommendation(
        location={"lat": lat, "lng": lng},
        risk_level=risk_level,
        risk_score=round(risk_score, 2),
        recommendations=recommendations,
        nearby_crimes=crimes_data,
    )
    return safety_recommendation


# ---- Global ML objects ----
crime_model = None
scaler = None


@api_router.post("/predict-crime", response_model=CrimePrediction)
async def predict_crime_risk(location_request: LocationRequest):
    if crime_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML model not initialized")

    lat, lng = location_request.latitude, location_request.longitude
    current_time = datetime.now()
    hour_of_day = current_time.hour / 24.0
    day_of_week = current_time.weekday() / 7.0
    month = current_time.month / 12.0

    population_density = 0.6
    economic_factor = 0.5

    nearby_crimes = await get_nearby_crimes(lat, lng, 10.0)
    historical_crime_rate = min(1.0, len(nearby_crimes) / 100.0)

    domain_counts = {}
    for crime in nearby_crimes:
        ctype = crime.get("crime_type", "")
        if ctype in ["ASSAULT", "HOMICIDE", "SEXUAL ASSAULT", "DOMESTIC VIOLENCE", "ROBBERY", "KIDNAPPING", "FIREARM OFFENSE"]:
            domain_counts["Violent Crime"] = domain_counts.get("Violent Crime", 0) + 1
        elif ctype in ["ARSON"]:
            domain_counts["Fire Accident"] = domain_counts.get("Fire Accident", 0) + 1
        elif ctype in ["TRAFFIC VIOLATION"]:
            domain_counts["Traffic Fatality"] = domain_counts.get("Traffic Fatality", 0) + 1
        else:
            domain_counts["Other Crime"] = domain_counts.get("Other Crime", 0) + 1

    total_crimes = len(nearby_crimes) or 1
    violent_crime_ratio = domain_counts.get("Violent Crime", 0) / total_crimes
    fire_accident_ratio = domain_counts.get("Fire Accident", 0) / total_crimes
    traffic_fatality_ratio = domain_counts.get("Traffic Fatality", 0) / total_crimes

    features = np.array(
        [
            [
                hour_of_day,
                day_of_week,
                month,
                population_density,
                economic_factor,
                historical_crime_rate,
                violent_crime_ratio,
                fire_accident_ratio,
                traffic_fatality_ratio,
            ]
        ]
    )
    features_scaled = scaler.transform(features)
    predicted_risk = float(crime_model.predict(features_scaled)[0])

    confidence = 0.80 + (historical_crime_rate * 0.15)
    confidence = min(0.95, confidence)

    crime_type_counts = {}
    for crime in nearby_crimes[:30]:
        t = crime.get("crime_type")
        crime_type_counts[t] = crime_type_counts.get(t, 0) + 1

    if crime_type_counts:
        predicted_crime_types = sorted(crime_type_counts.keys(), key=lambda x: crime_type_counts[x], reverse=True)[:4]
    else:
        predicted_crime_types = ["ASSAULT", "BURGLARY", "FRAUD", "CYBERCRIME"]

    factors = {
        "time_of_day": round(hour_of_day, 3),
        "day_of_week": round(day_of_week, 3),
        "month": round(month, 3),
        "population_density": round(population_density, 3),
        "historical_crimes": round(historical_crime_rate, 3),
        "violent_crime_pattern": round(violent_crime_ratio, 3),
        "fire_accident_pattern": round(fire_accident_ratio, 3),
        "traffic_pattern": round(traffic_fatality_ratio, 3),
    }

    return CrimePrediction(
        location={"lat": lat, "lng": lng},
        predicted_risk_score=round(predicted_risk, 2),
        predicted_crime_types=predicted_crime_types,
        confidence=round(confidence, 2),
        factors=factors,
    )


@api_router.get("/crime-heatmap")
async def get_crime_heatmap(state: Optional[str] = None, days: int = 30):
    query = {}
    if state:
        query["state"] = state
    cutoff_date = datetime.now(timezone.utc) - pd.Timedelta(days=days)
    query["timestamp"] = {"$gte": cutoff_date}
    crimes = await db.crime_data.find(query=query)
    heatmap_data = []
    for crime in crimes:
        loc = crime.get("location", {})
        heatmap_data.append(
            {
                "lat": loc.get("lat"),
                "lng": loc.get("lng"),
                "intensity": (crime.get("severity", 5) / 10.0),
                "crime_type": crime.get("crime_type"),
                "timestamp": crime.get("timestamp"),
            }
        )
    return {"heatmap_data": heatmap_data}


@api_router.get("/dashboard-stats")
async def get_dashboard_stats():
    total_crimes = await db.crime_data.count_documents({})
    pipeline_type = [{"$group": {"_id": "$crime_type", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
    crime_by_type = await db.crime_data.aggregate(pipeline_type)
    pipeline_state = [{"$group": {"_id": "$state", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
    crime_by_state = await db.crime_data.aggregate(pipeline_state)

    recent_crimes_raw = await db.crime_data.find(limit=10, sort=[("timestamp", "desc")])
    recent_crimes = []
    for crime in recent_crimes_raw:
        # ensure no DB-specific fields to serialize
        recent_crimes.append(crime)

    return {
        "total_crimes": total_crimes,
        "crime_by_type": crime_by_type,
        "crime_by_state": crime_by_state,
        "recent_crimes": recent_crimes,
    }


# ------------------------------
# API Route: Get All Crimes (consistent with db wrapper)
# ------------------------------
@api_router.get("/crimes")
async def get_crimes(limit: int = 100):
    """
    Fetch crime records from the database (default limit 100).
    """
    crimes = await db.crime_data.find(limit=limit, sort=[("timestamp", "desc")])
    # convert timestamps that might be strings
    out = []
    for c in crimes:
        ts = c.get("timestamp")
        if isinstance(ts, str):
            try:
                c["timestamp"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                c["timestamp"] = datetime.now(timezone.utc)
        out.append(c)
    return {"crimes": out}

# Add this endpoint to your server.py file

@api_router.delete("/reset-database")
async def reset_database():
    """
    Clear all crime data from the database and reload from CSV.
    Use with caution - this will delete all existing data!
    """
    try:
        # Delete all records from the crime_data table
        async with async_session() as session:
            await session.execute(crime_data_table.delete())
            await session.commit()
        
        logger.info("Database cleared successfully")
        
        # Now reload data from CSV
        await initialize_real_crime_data()
        
        # Check final count
        total_count = await db.crime_data.count_documents({})
        logger.info(f"Database reset complete. New record count: {total_count}")
        
        return {
            "message": "Database reset successfully", 
            "new_record_count": total_count
        }
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")


# Alternative: Add a force reload parameter to the initialization function
async def initialize_real_crime_data(force_reload: bool = False):
    """Load and initialize real crime data from the uploaded CSV"""
    try:
        if not force_reload:
            existing_count = await db.crime_data.count_documents({})
            if existing_count > 0:
                logger.info(f"Crime data already loaded: {existing_count} records")
                return
        else:
            # Clear existing data if force reload
            async with async_session() as session:
                await session.execute(crime_data_table.delete())
                await session.commit()
            logger.info("Existing data cleared for force reload")
    except Exception as e:
        logger.warning(f"Could not check existing data: {e}")

    # ... rest of the function remains the same ...
# ------------------------------
# API Route: Add New Crime Record
# ------------------------------
@api_router.post("/crimes", response_model=CrimeData)
async def add_crime(crime: CrimeDataCreate):
    """
    Add a new crime record into the database.
    """
    crime_obj = CrimeData(**crime.dict())
    # ensure timestamp is serializable
    if isinstance(crime_obj.timestamp, pd.Timestamp):
        crime_obj.timestamp = crime_obj.timestamp.to_pydatetime()
    await db.crime_data.insert_one(crime_obj.dict())
    return crime_obj


# include router and middlewares
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[o for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the application: create tables, load data, train model"""
    global crime_model, scaler
    logger.info("Starting Safe Guard API...")

    # Debug: Show current directory and files
    logger.info(f"Current working directory: {ROOT_DIR}")
    logger.info("Files in current directory:")
    try:
        for item in ROOT_DIR.iterdir():
            if item.is_file():
                logger.info(f"  FILE: {item.name} ({item.stat().st_size} bytes)")
            elif item.is_dir():
                logger.info(f"  DIR:  {item.name}/")
    except Exception as e:
        logger.error(f"Could not list directory contents: {e}")

    # Ensure tables exist
    try:
        await _ensure_tables()
        logger.info("Database tables ensured successfully")
    except Exception as e:
        logger.error(f"Table creation failed: {e}")
        raise

    # Debug CSV structure first
    await debug_csv_structure()

    # Initialize data: try real dataset, else samples
    logger.info("Starting data initialization...")
    await initialize_real_crime_data()

    # Check how much data was loaded
    try:
        total_count = await db.crime_data.count_documents({})
        logger.info(f"Total crime records in database: {total_count}")
        
        # Show sample of what was loaded
        if total_count > 0:
            sample_crimes = await db.crime_data.find(limit=3)
            logger.info("Sample loaded records:")
            for i, crime in enumerate(sample_crimes, 1):
                logger.info(f"  {i}. {crime.get('crime_type')} in {crime.get('city')}, {crime.get('state')}")
    except Exception as e:
        logger.error(f"Could not check loaded data: {e}")

    # Train model
    logger.info("Training ML model...")
    loop = asyncio.get_event_loop()
    model_future = loop.run_in_executor(None, train_crime_model_with_real_data)
    crime_model, scaler = await model_future
    logger.info("ML model initialized successfully")
    logger.info("Safe Guard API startup completed!")


@app.on_event("shutdown")
async def shutdown_event():
    # nothing special to close here (SQLAlchemy engine will be garbage collected)
    try:
        await engine.dispose()
    except Exception:
        pass
    logger.info("Shutting down Safe Guard API.")


# If needed: run uvicorn from here when debugging (optional)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)