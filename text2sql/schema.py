"""The prompt the model sees: schema, capital ids, today's date and the question.

The same prompt is used for training, for evaluating the base model (baseline) and in the browser demo,
so the comparison is fair and the demo behaves like the evaluation.
"""
from __future__ import annotations

SYSTEM = (
    "You translate questions about the weather in Spain's provincial capitals into one DuckDB SQL query "
    "over the tables below. Reply with the SQL query only."
)

SCHEMA = """\
CREATE TABLE latest_forecast ( -- newest 72-hour forecast, one row per capital and hour
  location_id VARCHAR, location_name VARCHAR, province VARCHAR, latitude DOUBLE, longitude DOUBLE,
  issued_at TIMESTAMP, valid_at TIMESTAMP, lead_hours BIGINT, temperature_c DOUBLE,
  relative_humidity_pct DOUBLE, dew_point_c DOUBLE, precipitation_mm DOUBLE, pressure_msl_hpa DOUBLE,
  cloud_cover_pct DOUBLE, wind_speed_kmh DOUBLE, wind_direction_deg DOUBLE, wind_gusts_kmh DOUBLE,
  shortwave_radiation_wm2 DOUBLE);
CREATE TABLE location_station ( -- nearest AEMET weather station to each capital
  location_id VARCHAR, location_name VARCHAR, province VARCHAR, latitude DOUBLE, longitude DOUBLE,
  station_id VARCHAR, station_name VARCHAR, station_altitude_m DOUBLE, distance_km DOUBLE);
CREATE TABLE daily_summary ( -- observed weather per capital and day, and forecast error by lead time
  location_id VARCHAR, date DATE, observed_hours BIGINT, temperature_min_c DOUBLE, temperature_max_c DOUBLE,
  temperature_mean_c DOUBLE, precipitation_sum_mm DOUBLE, wind_speed_max_kmh DOUBLE,
  temperature_mae_lead_1_6h DOUBLE, temperature_mae_lead_7_24h DOUBLE, temperature_mae_lead_25_48h DOUBLE,
  temperature_mae_lead_49_72h DOUBLE);
CREATE TABLE forecast_vs_observed ( -- every forecast made for an observed hour; error = forecast - observed
  location_id VARCHAR, station_id VARCHAR, date DATE, valid_at TIMESTAMP, issued_at TIMESTAMP,
  lead_hours BIGINT, forecast_temperature_c DOUBLE, observed_temperature_c DOUBLE,
  error_temperature_c DOUBLE, forecast_relative_humidity_pct DOUBLE, observed_relative_humidity_pct DOUBLE,
  error_relative_humidity_pct DOUBLE, forecast_precipitation_mm DOUBLE, observed_precipitation_mm DOUBLE,
  forecast_pressure_msl_hpa DOUBLE, observed_pressure_msl_hpa DOUBLE, error_pressure_msl_hpa DOUBLE,
  forecast_wind_speed_kmh DOUBLE, observed_wind_speed_kmh DOUBLE, error_wind_speed_kmh DOUBLE);
-- All timestamps are UTC."""

# location_id -> names people use for the capital (first = official name)
CAPITALS: dict[str, list[str]] = {
    "a-coruna": ["A Coruña", "La Coruña", "Coruña"],
    "albacete": ["Albacete"],
    "alicante": ["Alicante", "Alacant"],
    "almeria": ["Almería"],
    "avila": ["Ávila"],
    "badajoz": ["Badajoz"],
    "barcelona": ["Barcelona"],
    "bilbao": ["Bilbao"],
    "burgos": ["Burgos"],
    "caceres": ["Cáceres"],
    "cadiz": ["Cádiz"],
    "castellon-de-la-plana": ["Castellón de la Plana", "Castellón"],
    "ceuta": ["Ceuta"],
    "ciudad-real": ["Ciudad Real"],
    "cordoba": ["Córdoba"],
    "cuenca": ["Cuenca"],
    "girona": ["Girona", "Gerona"],
    "granada": ["Granada"],
    "guadalajara": ["Guadalajara"],
    "huelva": ["Huelva"],
    "huesca": ["Huesca"],
    "jaen": ["Jaén"],
    "las-palmas-de-gran-canaria": ["Las Palmas de Gran Canaria", "Las Palmas"],
    "leon": ["León"],
    "lleida": ["Lleida", "Lérida"],
    "logrono": ["Logroño"],
    "lugo": ["Lugo"],
    "madrid": ["Madrid"],
    "malaga": ["Málaga"],
    "melilla": ["Melilla"],
    "murcia": ["Murcia"],
    "ourense": ["Ourense", "Orense"],
    "oviedo": ["Oviedo"],
    "palencia": ["Palencia"],
    "palma": ["Palma", "Palma de Mallorca"],
    "pamplona": ["Pamplona", "Iruña"],
    "pontevedra": ["Pontevedra"],
    "salamanca": ["Salamanca"],
    "san-sebastian": ["San Sebastián", "Donostia"],
    "santa-cruz-de-tenerife": ["Santa Cruz de Tenerife"],
    "santander": ["Santander"],
    "segovia": ["Segovia"],
    "sevilla": ["Sevilla", "Seville"],
    "soria": ["Soria"],
    "tarragona": ["Tarragona"],
    "teruel": ["Teruel"],
    "toledo": ["Toledo"],
    "valencia": ["Valencia", "València"],
    "valladolid": ["Valladolid"],
    "vitoria-gasteiz": ["Vitoria-Gasteiz", "Vitoria"],
    "zamora": ["Zamora"],
    "zaragoza": ["Zaragoza"],
}


def user_prompt(question: str, today: str) -> str:
    ids = ", ".join(CAPITALS)
    return f"{SCHEMA}\n-- location_id values: {ids}\n\nToday (UTC): {today}\nQuestion: {question}"


def messages(question: str, today: str, sql: str | None = None) -> list[dict[str, str]]:
    """Chat messages for one example; with `sql` it is a training example, without it a request."""
    chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(question, today)}]
    if sql is not None:
        chat.append({"role": "assistant", "content": sql})
    return chat
