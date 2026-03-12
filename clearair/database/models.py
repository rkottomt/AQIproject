"""SQLAlchemy ORM models for the ClearAir database."""

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
    Index,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AirQualityReading(Base):
    """Stores individual pollutant measurements from various sources."""

    __tablename__ = "air_quality_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(64), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    datetime = Column(DateTime(timezone=True), nullable=False)
    parameter = Column(String(32), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(16), nullable=False)
    source = Column(String(32), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_aq_city_datetime", "city", "datetime"),
    )


class WeatherReading(Base):
    """Stores meteorological observations and reanalysis data."""

    __tablename__ = "weather_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(64), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    datetime = Column(DateTime(timezone=True), nullable=False)
    temperature_c = Column(Float)
    humidity_pct = Column(Float)
    wind_speed_ms = Column(Float)
    wind_direction_deg = Column(Float)
    precipitation_mm = Column(Float)
    boundary_layer_height_m = Column(Float)
    uv_index = Column(Float)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_weather_city_datetime", "city", "datetime"),
    )


class TrafficReading(Base):
    """Stores traffic flow and congestion metrics."""

    __tablename__ = "traffic_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(64), nullable=False)
    datetime = Column(DateTime(timezone=True), nullable=False)
    congestion_index = Column(Float)
    peak_hour_flag = Column(Boolean)
    avg_speed_kmh = Column(Float)
    traffic_volume = Column(Float)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_traffic_city_datetime", "city", "datetime"),
    )


class AQIForecast(Base):
    """Stores model-generated AQI forecasts with quantile predictions."""

    __tablename__ = "aqi_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(64), nullable=False)
    forecast_datetime = Column(DateTime(timezone=True), nullable=False)
    horizon_hours = Column(Integer, nullable=False)
    aqi_p10 = Column(Float)
    aqi_p50 = Column(Float)
    aqi_p90 = Column(Float)
    model_version = Column(String(64))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_forecast_city_datetime", "city", "forecast_datetime"),
    )


class CountermeasureEvent(Base):
    """Records historical countermeasure deployments and their measured effects."""

    __tablename__ = "countermeasure_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(64), nullable=False)
    countermeasure_key = Column(String(64), nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    aqi_before = Column(Float)
    aqi_after = Column(Float)
    pct_change = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_cm_city_datetime", "city", "start_date"),
    )
