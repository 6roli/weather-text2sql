"""Question families: each one writes a question (Spanish or English) and the SQL that answers it.

Every family is a function `(Ctx) -> (question, sql)`. All randomness comes from `ctx.rng`, so calling a
family twice with the same seed gives the same example; only `ctx.shift` changes, which moves every date
by that many days. The generator uses this to validate an example on the real snapshot dates (shift 0)
and then train on shifted dates, so the model learns to read "Today" from the prompt instead of
memorising the snapshot's date.

SQL conventions the model learns:
- relative days are written from today's date: DATE '2026-10-25' + INTERVAL 1 DAY ("tomorrow");
- a calendar day of a timestamp is CAST(valid_at AS DATE);
- rankings break ties by name (ORDER BY value DESC, location_name), so results are deterministic;
- "the next N hours" of the forecast is lead_hours <= N.
"""
from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from .schema import CAPITALS

FORECAST_DAY = date(2026, 10, 25)  # the snapshot's latest_forecast was issued on this day
FIRST_OBSERVED = date(2026, 10, 18)  # first day in daily_summary / forecast_vs_observed
LAST_OBSERVED = date(2026, 10, 25)

MONTHS_ES = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
MONTHS_EN = "January February March April May June July August September October November December".split()


@dataclass
class Ctx:
    rng: random.Random
    real_today: date
    shift: int = 0
    lang: str = "es"

    @property
    def today(self) -> date:
        return self.real_today + timedelta(days=self.shift)

    def pick(self, es: list[str], en: list[str]) -> str:
        return self.rng.choice(es if self.lang == "es" else en)

    def city(self, exclude: str | None = None) -> tuple[str, str]:
        location_id = self.rng.choice([c for c in CAPITALS if c != exclude])
        name = self.rng.choice(CAPITALS[location_id])
        roll = self.rng.random()
        if roll < 0.10:
            name = strip_accents(name).lower()  # people type "cordoba", "a coruna"
        elif roll < 0.15:
            name = name.lower()
        return location_id, name


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def day_expr(today: date, offset: int) -> str:
    base = f"DATE '{today.isoformat()}'"
    if offset == 0:
        return base
    return f"{base} {'+' if offset > 0 else '-'} INTERVAL {abs(offset)} DAY"


def date_literal(day: date) -> str:
    return f"DATE '{day.isoformat()}'"


def hour_expr(day_sql: str, hour: int) -> str:
    return f"{day_sql} + INTERVAL {hour} HOUR" if hour else day_sql


# ---------------------------------------------------------------- day phrases

FORECAST_DAYS = [  # offset from today, Spanish, English
    (0, ["hoy"], ["today"]),
    (1, ["mañana"], ["tomorrow"]),
    (2, ["pasado mañana"], ["the day after tomorrow"]),
]


def forecast_day(ctx: Ctx) -> tuple[str, str]:
    """A day covered by the forecast: (phrase, SQL date expression)."""
    offset, es, en = ctx.rng.choice(FORECAST_DAYS)
    return ctx.pick(es, en), day_expr(ctx.today, offset)


def explicit_date(ctx: Ctx, day: date) -> str:
    if ctx.lang == "es":
        options = [f"el {day.day} de {MONTHS_ES[day.month - 1]}", f"el {day.day}/{day.month}", f"el {day.isoformat()}", f"el día {day.day}"]
    else:
        options = [f"on {MONTHS_EN[day.month - 1]} {day.day}", f"on {day.day} {MONTHS_EN[day.month - 1]}", f"on {day.isoformat()}"]
    # always one draw from a fixed-length list, so shifting dates never changes the rest of the example
    index = ctx.rng.randrange(len(options))
    if index == 3 and day.month != ctx.today.month:  # "el día 28" only makes sense within the current month
        index = 0
    return options[index]


def observed_day(ctx: Ctx) -> tuple[str, str]:
    """A past day with observations: relative ("ayer") or a calendar date; (phrase, SQL date expression)."""
    real_days = (ctx.real_today - FIRST_OBSERVED).days
    if ctx.rng.random() < 0.55:
        offset, es, en = ctx.rng.choice([
            (0, ["hoy"], ["today"]),
            (1, ["ayer"], ["yesterday"]),
            (1, ["ayer"], ["yesterday"]),
            (2, ["anteayer", "antes de ayer"], ["the day before yesterday"]),
        ])
        if offset <= real_days:
            return ctx.pick(es, en), day_expr(ctx.today, -offset)
    back = ctx.rng.randint(0, real_days)
    day = ctx.today - timedelta(days=back)
    return explicit_date(ctx, day), date_literal(day)


def since_date(ctx: Ctx) -> tuple[str, str]:
    back = ctx.rng.randint(1, (ctx.real_today - FIRST_OBSERVED).days)
    day = ctx.today - timedelta(days=back)
    if ctx.lang == "es":
        phrase = ctx.rng.choice([f"desde el {day.day} de {MONTHS_ES[day.month - 1]}", f"desde el {day.isoformat()}"])
    else:
        phrase = ctx.rng.choice([f"since {MONTHS_EN[day.month - 1]} {day.day}", f"since {day.isoformat()}"])
    return phrase, date_literal(day)


def last_days(ctx: Ctx) -> tuple[str, int]:
    n = ctx.rng.choice([2, 3, 3, 5, 7, 7])
    return ctx.pick([f"en los últimos {n} días", f"los últimos {n} días"], [f"in the last {n} days", f"over the last {n} days"]), n


# ---------------------------------------------------------------- variables

@dataclass(frozen=True)
class Var:
    column: str
    es: str  # with article
    es_max: str
    es_min: str
    es_avg: str
    feminine: bool
    en: str
    unit: str


FORECAST_VARS = [
    Var("temperature_c", "la temperatura", "la temperatura máxima", "la temperatura mínima", "la temperatura media", True, "temperature", "°C"),
    Var("relative_humidity_pct", "la humedad", "la humedad máxima", "la humedad mínima", "la humedad media", True, "humidity", "%"),
    Var("wind_speed_kmh", "la velocidad del viento", "el viento máximo", "el viento mínimo", "el viento medio", False, "wind speed", "km/h"),
    Var("wind_gusts_kmh", "las rachas de viento", "la racha máxima de viento", "la racha mínima de viento", "la racha media de viento", True, "wind gusts", "km/h"),
    Var("pressure_msl_hpa", "la presión", "la presión máxima", "la presión mínima", "la presión media", True, "pressure", "hPa"),
    Var("cloud_cover_pct", "la nubosidad", "la nubosidad máxima", "la nubosidad mínima", "la nubosidad media", True, "cloud cover", "%"),
    Var("dew_point_c", "el punto de rocío", "el punto de rocío máximo", "el punto de rocío mínimo", "el punto de rocío medio", False, "dew point", "°C"),
    Var("shortwave_radiation_wm2", "la radiación solar", "la radiación solar máxima", "la radiación solar mínima", "la radiación solar media", True, "solar radiation", "W/m²"),
]
TEMPERATURE = FORECAST_VARS[0]


def forecast_var(ctx: Ctx, temperature_weight: float = 0.4) -> Var:
    return TEMPERATURE if ctx.rng.random() < temperature_weight else ctx.rng.choice(FORECAST_VARS[1:])


# ================================================================ latest_forecast

def fc_value_at_hour(ctx: Ctx):
    location_id, city = ctx.city()
    var = forecast_var(ctx)
    day, day_sql = forecast_day(ctx)
    hour = ctx.rng.randint(0, 23)
    clock = ctx.rng.choice([f"{hour}:00", f"{hour:02d}:00", f"{hour}h"])
    q = ctx.pick(
        [f"¿Qué {var.es.split(' ', 1)[1]} habrá en {city} {day} a las {clock}?",
         f"¿Cuál será {var.es} en {city} {day} a las {clock} UTC?",
         f"Previsión de {var.es.split(' ', 1)[1]} en {city} {day} a las {clock}",
         f"Dime {var.es} prevista para {city} {day} a las {clock}" if var.feminine and not var.es.startswith("las") else f"Dime {var.es} en {city} {day} a las {clock}"],
        [f"What will the {var.en} be in {city} {day} at {clock}?",
         f"Forecast {var.en} for {city} {day} at {clock} UTC"],
    )
    sql = f"SELECT {var.column} FROM latest_forecast WHERE location_id = '{location_id}' AND valid_at = {hour_expr(day_sql, hour)}"
    return q, sql


def fc_stat_day(ctx: Ctx):
    location_id, city = ctx.city()
    var = forecast_var(ctx, 0.6)
    day, day_sql = forecast_day(ctx)
    func, es, en = ctx.rng.choice([("max", var.es_max, f"maximum {var.en}"), ("min", var.es_min, f"minimum {var.en}"), ("avg", var.es_avg, f"average {var.en}")])
    q = ctx.pick(
        [f"¿Cuál será {es} en {city} {day}?", f"Previsión de {es.split(' ', 1)[1]} para {city} {day}", f"¿Qué {es.split(' ', 1)[1]} se espera en {city} {day}?",
         f"Dime {es} de {day} en {city}"],
        [f"What is the {en} forecast for {city} {day}?", f"{en.capitalize()} in {city} {day}"],
    )
    sql = f"SELECT {func}({var.column}) FROM latest_forecast WHERE location_id = '{location_id}' AND CAST(valid_at AS DATE) = {day_sql}"
    return q, sql


def fc_rain_day(ctx: Ctx):
    location_id, city = ctx.city()
    day, day_sql = forecast_day(ctx)
    q = ctx.pick(
        [f"¿Cuánto va a llover en {city} {day}?", f"¿Cuánta lluvia se espera en {city} {day}?", f"Precipitación total prevista en {city} {day}",
         f"¿Cuántos litros caerán en {city} {day}?", f"¿Lloverá {day} en {city}?"],
        [f"How much rain is expected in {city} {day}?", f"Total precipitation forecast for {city} {day}", f"Will it rain in {city} {day}?"],
    )
    sql = f"SELECT sum(precipitation_mm) FROM latest_forecast WHERE location_id = '{location_id}' AND CAST(valid_at AS DATE) = {day_sql}"
    return q, sql


def fc_rain_next_hours(ctx: Ctx):
    location_id, city = ctx.city()
    hours = ctx.rng.choice([6, 12, 24, 24, 48, 72])
    q = ctx.pick(
        [f"¿Va a llover en {city} en las próximas {hours} horas?", f"Lluvia prevista en {city} para las próximas {hours} horas",
         f"¿Cuánto lloverá en {city} en las próximas {hours} h?"],
        [f"Will it rain in {city} in the next {hours} hours?", f"Rain expected in {city} over the next {hours} hours"],
    )
    sql = f"SELECT sum(precipitation_mm) FROM latest_forecast WHERE location_id = '{location_id}' AND lead_hours <= {hours}"
    return q, sql


def fc_ranking(ctx: Ctx):
    """Top / bottom N capitals for a day (or the whole forecast)."""
    n = ctx.rng.choice([1, 3, 3, 5, 5, 10])
    whole = ctx.rng.random() < 0.25
    day, day_sql = ("", "") if whole else forecast_day(ctx)
    when_es = ctx.rng.choice(["en los próximos días", "en los próximos 3 días"]) if whole else day
    when_en = "over the next days" if whole else day
    kind = ctx.rng.choice(["hot", "hot", "cold", "rain", "rain", "wind"])
    if kind == "hot":
        expr, alias, order = "max(temperature_c)", "temperature_max_c", "DESC"
        es = [f"las {n} capitales más calurosas {when_es}", f"las {n} capitales con la temperatura máxima más alta {when_es}"]
        en = [f"the {n} hottest capitals {when_en}"]
        es1, en1 = [f"la capital más calurosa {when_es}", f"dónde hará más calor {when_es}"], [f"the hottest capital {when_en}"]
    elif kind == "cold":
        expr, alias, order = "min(temperature_c)", "temperature_min_c", "ASC"
        es = [f"las {n} capitales más frías {when_es}", f"las {n} capitales con la mínima más baja {when_es}"]
        en = [f"the {n} coldest capitals {when_en}"]
        es1, en1 = [f"la capital más fría {when_es}", f"dónde hará más frío {when_es}"], [f"the coldest capital {when_en}"]
    elif kind == "rain":
        expr, alias, order = "sum(precipitation_mm)", "precipitation_mm", "DESC"
        es = [f"las {n} capitales donde más lloverá {when_es}", f"las {n} capitales con más precipitación prevista {when_es}"]
        en = [f"the {n} capitals with the most rain {when_en}"]
        es1, en1 = [f"la capital donde más lloverá {when_es}", f"dónde lloverá más {when_es}"], [f"the capital with the most rain {when_en}"]
    else:
        expr, alias, order = "max(wind_gusts_kmh)", "wind_gusts_max_kmh", "DESC"
        es = [f"las {n} capitales con rachas de viento más fuertes {when_es}", f"las {n} capitales más ventosas {when_es}"]
        en = [f"the {n} capitals with the strongest gusts {when_en}"]
        es1, en1 = [f"la capital con las rachas más fuertes {when_es}", f"dónde soplará más el viento {when_es}"], [f"the windiest capital {when_en}"]
    if n == 1:
        phrase = ctx.pick(es1, en1)
    else:
        phrase = ctx.pick(es, en)
    q = ctx.pick([f"¿Cuáles son {phrase}?", f"Dime {phrase}", f"{phrase[0].upper()}{phrase[1:]}"] if n > 1 else [f"¿Cuál es {phrase}?", f"Dime {phrase}", f"¿{phrase[0].upper()}{phrase[1:]}?"],
                 [f"Which are {phrase}?", f"List {phrase}"] if n > 1 else [f"Which is {phrase}?", f"What is {phrase}?"])
    where = "" if whole else f" WHERE CAST(valid_at AS DATE) = {day_sql}"
    sql = (f"SELECT location_name, {expr} AS {alias} FROM latest_forecast{where} "
           f"GROUP BY location_name ORDER BY {alias} {order}, location_name LIMIT {n}")
    return q, sql


def fc_threshold(ctx: Ctx):
    day, day_sql = forecast_day(ctx)
    kind = ctx.rng.choice(["hot", "frost", "rain", "gusts", "cold"])
    if kind == "hot":
        x = ctx.rng.choice([20, 22, 25, 28])
        cond, es, en = f"temperature_c > {x}", f"se superarán los {x} °C", f"temperatures above {x} °C"
    elif kind == "cold":
        x = ctx.rng.choice([5, 8, 10])
        cond, es, en = f"temperature_c < {x}", f"bajará de {x} °C", f"temperatures below {x} °C"
    elif kind == "frost":
        cond, es, en = "temperature_c < 0", "helará", "frost (below 0 °C)"
    elif kind == "rain":
        cond, es, en = "precipitation_mm > 0", "lloverá", "rain"
    else:
        x = ctx.rng.choice([30, 40, 50, 60])
        cond, es, en = f"wind_gusts_kmh > {x}", f"habrá rachas de más de {x} km/h", f"wind gusts above {x} km/h"
    if ctx.rng.random() < 0.5:
        q = ctx.pick([f"¿En cuántas capitales {es} {day}?", f"Número de capitales donde {es} {day}"],
                     [f"In how many capitals will there be {en} {day}?"])
        sql = f"SELECT count(DISTINCT location_id) FROM latest_forecast WHERE CAST(valid_at AS DATE) = {day_sql} AND {cond}"
    else:
        q = ctx.pick([f"¿En qué capitales {es} {day}?", f"¿Dónde {es} {day}?", f"Lista de capitales donde {es} {day}"],
                     [f"Which capitals will have {en} {day}?", f"Where will there be {en} {day}?"])
        sql = f"SELECT DISTINCT location_name FROM latest_forecast WHERE CAST(valid_at AS DATE) = {day_sql} AND {cond}"
    return q, sql


def fc_hourly(ctx: Ctx):
    location_id, city = ctx.city()
    day, day_sql = forecast_day(ctx)
    if ctx.rng.random() < 0.5:
        q = ctx.pick([f"Previsión hora a hora para {city} {day}", f"Dame el tiempo por horas en {city} {day}", f"¿Qué tiempo hará en {city} {day}, por horas?"],
                     [f"Hourly forecast for {city} {day}", f"Hour by hour weather in {city} {day}"])
        cols = "valid_at, temperature_c, precipitation_mm, wind_speed_kmh, cloud_cover_pct"
    else:
        var = forecast_var(ctx, 0.5)
        q = ctx.pick([f"{var.es.capitalize()} hora a hora en {city} {day}", f"¿Cómo evolucionará {var.es} en {city} {day}?"],
                     [f"Hourly {var.en} in {city} {day}", f"How will the {var.en} change in {city} {day}?"])
        cols = f"valid_at, {var.column}"
    sql = f"SELECT {cols} FROM latest_forecast WHERE location_id = '{location_id}' AND CAST(valid_at AS DATE) = {day_sql} ORDER BY valid_at"
    return q, sql


def fc_when_peak(ctx: Ctx):
    location_id, city = ctx.city()
    whole = ctx.rng.random() < 0.4
    day, day_sql = ("", "") if whole else forecast_day(ctx)
    when_es = "en los próximos días" if whole else day
    when_en = "in the next days" if whole else day
    kind = ctx.rng.choice(["hot", "cold", "gusts", "rain"])
    order, column, es, en = {
        "hot": ("DESC", "temperature_c", f"¿A qué hora hará más calor en {city} {when_es}?", f"At what time will it be hottest in {city} {when_en}?"),
        "cold": ("ASC", "temperature_c", f"¿Cuándo hará más frío en {city} {when_es}?", f"When will it be coldest in {city} {when_en}?"),
        "gusts": ("DESC", "wind_gusts_kmh", f"¿Cuándo soplará más fuerte el viento en {city} {when_es}?", f"When will the strongest gusts hit {city} {when_en}?"),
        "rain": ("DESC", "precipitation_mm", f"¿A qué hora lloverá más en {city} {when_es}?", f"At what time will it rain the most in {city} {when_en}?"),
    }[kind]
    q = ctx.pick([es], [en])
    where = f"location_id = '{location_id}'" + ("" if whole else f" AND CAST(valid_at AS DATE) = {day_sql}")
    sql = f"SELECT valid_at, {column} FROM latest_forecast WHERE {where} ORDER BY {column} {order}, valid_at LIMIT 1"
    return q, sql


def fc_compare(ctx: Ctx):
    a_id, a = ctx.city()
    b_id, b = ctx.city(exclude=a_id)
    day, day_sql = forecast_day(ctx)
    if ctx.rng.random() < 0.6:
        q = ctx.pick([f"¿Dónde hará más calor {day}, en {a} o en {b}?", f"Compara la temperatura máxima de {a} y {b} {day}", f"¿Hará más calor {day} en {a} que en {b}?"],
                     [f"Where will it be warmer {day}, {a} or {b}?", f"Compare the maximum temperature in {a} and {b} {day}"])
        expr, alias = "max(temperature_c)", "temperature_max_c"
    else:
        q = ctx.pick([f"¿Dónde lloverá más {day}, en {a} o en {b}?", f"Compara la lluvia prevista {day} en {a} y en {b}"],
                     [f"Where will it rain more {day}, {a} or {b}?", f"Compare tomorrow's rain in {a} and {b}" if day == "tomorrow" else f"Compare the rain in {a} and {b} {day}"])
        expr, alias = "sum(precipitation_mm)", "precipitation_mm"
    sql = (f"SELECT location_name, {expr} AS {alias} FROM latest_forecast WHERE location_id IN ('{a_id}', '{b_id}') "
           f"AND CAST(valid_at AS DATE) = {day_sql} GROUP BY location_name ORDER BY {alias} DESC")
    return q, sql


def fc_daily_table(ctx: Ctx):
    location_id, city = ctx.city()
    if ctx.rng.random() < 0.6:
        q = ctx.pick([f"Máximas y mínimas previstas en {city} para cada día", f"Temperatura máxima y mínima por día en {city} según la previsión",
                      f"¿Qué máximas y mínimas tendrá {city} los próximos días?"],
                     [f"Daily minimum and maximum temperature forecast for {city}", f"Highs and lows in {city} for the coming days"])
        cols = "min(temperature_c) AS temperature_min_c, max(temperature_c) AS temperature_max_c"
    else:
        q = ctx.pick([f"Lluvia prevista por día en {city}", f"¿Cuánto lloverá cada día en {city}?"],
                     [f"Daily rain forecast for {city}", f"How much rain each day in {city}?"])
        cols = "sum(precipitation_mm) AS precipitation_mm"
    sql = (f"SELECT CAST(valid_at AS DATE) AS day, {cols} FROM latest_forecast WHERE location_id = '{location_id}' "
           f"GROUP BY day ORDER BY day")
    return q, sql


def fc_issued(ctx: Ctx):
    q = ctx.pick(["¿Cuándo se emitió la última previsión?", "¿De qué hora es la previsión más reciente?", "¿Cuándo se actualizó la previsión por última vez?"],
                 ["When was the latest forecast issued?", "How recent is the forecast?"])
    return q, "SELECT max(issued_at) FROM latest_forecast"


# ================================================================ location_station

def st_station_of(ctx: Ctx):
    location_id, city = ctx.city()
    q = ctx.pick([f"¿Qué estación de AEMET se usa para {city}?", f"¿Cuál es la estación meteorológica más cercana a {city}?", f"Estación de observación de {city} y su distancia"],
                 [f"Which AEMET station is used for {city}?", f"Nearest weather station to {city}"])
    return q, f"SELECT station_id, station_name, distance_km FROM location_station WHERE location_id = '{location_id}'"


def st_attribute(ctx: Ctx):
    location_id, city = ctx.city()
    cols, es, en = ctx.rng.choice([
        ("latitude, longitude", [f"Coordenadas de {city}", f"¿Cuál es la latitud y longitud de {city}?"], [f"Coordinates of {city}"]),
        ("province", [f"¿En qué provincia está {city}?", f"Provincia de {city}"], [f"Which province is {city} in?"]),
        ("station_altitude_m", [f"¿A qué altitud está la estación de {city}?", f"Altitud de la estación meteorológica de {city}"], [f"Altitude of the weather station of {city}"]),
        ("distance_km", [f"¿A qué distancia está la estación de AEMET de {city}?", f"¿Cuántos km hay de {city} a su estación?"], [f"How far is the station from {city}?"]),
    ])
    return ctx.pick(es, en), f"SELECT {cols} FROM location_station WHERE location_id = '{location_id}'"


def st_ranking(ctx: Ctx):
    n = ctx.rng.choice([1, 3, 5, 10])
    column, order, es, en = ctx.rng.choice([
        ("station_altitude_m", "DESC", "las estaciones más altas", "the highest stations"),
        ("station_altitude_m", "ASC", "las estaciones más bajas", "the lowest stations"),
        ("distance_km", "DESC", "las capitales con la estación más lejana", "the capitals whose station is farthest"),
        ("latitude", "DESC", "las capitales más al norte", "the northernmost capitals"),
        ("latitude", "ASC", "las capitales más al sur", "the southernmost capitals"),
        ("longitude", "ASC", "las capitales más al oeste", "the westernmost capitals"),
        ("longitude", "DESC", "las capitales más al este", "the easternmost capitals"),
    ])
    q = ctx.pick([f"¿Cuáles son las {n} {es.split(' ', 1)[1]}?", f"Top {n} de {es}"], [f"Top {n} {en}", f"Which are the {n} {en.split(' ', 1)[1]}?"])
    return q, f"SELECT location_name, {column} FROM location_station ORDER BY {column} {order}, location_name LIMIT {n}"


def st_threshold(ctx: Ctx):
    if ctx.rng.random() < 0.5:
        x = ctx.rng.choice([5, 10, 15, 20])
        q = ctx.pick([f"¿Qué capitales tienen la estación a más de {x} km?", f"Capitales cuya estación está a más de {x} km"],
                     [f"Which capitals have their station more than {x} km away?"])
        return q, f"SELECT location_name, station_name, distance_km FROM location_station WHERE distance_km > {x} ORDER BY distance_km DESC"
    x = ctx.rng.choice([500, 700, 800, 1000])
    q = ctx.pick([f"¿Qué estaciones están por encima de {x} metros?", f"Capitales con estación a más de {x} m de altitud"],
                 [f"Which stations are above {x} m?"])
    return q, f"SELECT location_name, station_name, station_altitude_m FROM location_station WHERE station_altitude_m > {x} ORDER BY station_altitude_m DESC"


# ================================================================ daily_summary

OBSERVED_STATS = [  # column, Spanish, English
    ("temperature_max_c", "la temperatura máxima", "maximum temperature"),
    ("temperature_min_c", "la temperatura mínima", "minimum temperature"),
    ("temperature_mean_c", "la temperatura media", "mean temperature"),
    ("precipitation_sum_mm", "la precipitación total", "total precipitation"),
    ("wind_speed_max_kmh", "el viento máximo", "maximum wind speed"),
]


def ds_value(ctx: Ctx):
    location_id, city = ctx.city()
    day, day_sql = observed_day(ctx)
    column, es, en = ctx.rng.choice(OBSERVED_STATS)
    if column == "precipitation_sum_mm" and ctx.rng.random() < 0.6:
        q = ctx.pick([f"¿Cuánto llovió en {city} {day}?", f"¿Llovió en {city} {day}?", f"Lluvia registrada en {city} {day}"],
                     [f"How much did it rain in {city} {day}?", f"Did it rain in {city} {day}?"])
    else:
        q = ctx.pick([f"¿Cuál fue {es} en {city} {day}?", f"¿Qué {es.split(' ', 1)[1]} se registró en {city} {day}?", f"Valor observado de {es} en {city} {day}"],
                     [f"What was the {en} in {city} {day}?", f"Observed {en} in {city} {day}"])
    return q, f"SELECT {column} FROM daily_summary WHERE location_id = '{location_id}' AND date = {day_sql}"


def ds_all_stats(ctx: Ctx):
    location_id, city = ctx.city()
    day, day_sql = observed_day(ctx)
    q = ctx.pick([f"¿Qué tiempo hizo en {city} {day}?", f"Resumen del tiempo en {city} {day}", f"Datos observados en {city} {day}"],
                 [f"What was the weather like in {city} {day}?", f"Weather summary for {city} {day}"])
    sql = (f"SELECT temperature_min_c, temperature_max_c, temperature_mean_c, precipitation_sum_mm, wind_speed_max_kmh "
           f"FROM daily_summary WHERE location_id = '{location_id}' AND date = {day_sql}")
    return q, sql


def ds_ranking(ctx: Ctx):
    day, day_sql = observed_day(ctx)
    n = ctx.rng.choice([1, 3, 5, 5, 10])
    expr, alias, order, es_one, es_many, en = ctx.rng.choice([
        ("temperature_max_c", "temperature_max_c", "DESC", "la más calurosa", "más calurosas", "hottest"),
        ("temperature_min_c", "temperature_min_c", "ASC", "la más fría", "más frías", "coldest"),
        ("precipitation_sum_mm", "precipitation_sum_mm", "DESC", "la más lluviosa", "más lluviosas", "rainiest"),
        ("wind_speed_max_kmh", "wind_speed_max_kmh", "DESC", "la de más viento", "con más viento", "windiest"),
        ("temperature_max_c - temperature_min_c", "temperature_range_c", "DESC", "la de mayor amplitud térmica", "con mayor amplitud térmica", "largest temperature range"),
    ])
    if n == 1:
        q = ctx.pick([f"¿Qué capital fue {es_one} {day}?", f"¿Cuál fue la capital {es_one.split(' ', 1)[1]} {day}?"],
                     [f"Which capital had the {en} {day}?"])
    else:
        q = ctx.pick([f"¿Cuáles fueron las {n} capitales {es_many} {day}?", f"Las {n} capitales {es_many} {day}"],
                     [f"Which {n} capitals had the {en} {day}?"])
    select = expr if expr == alias else f"{expr} AS {alias}"
    sql = (f"SELECT location_id, {select} FROM daily_summary WHERE date = {day_sql} AND {expr} IS NOT NULL "
           f"ORDER BY {alias} {order}, location_id LIMIT {n}")
    return q, sql


def ds_period_total(ctx: Ctx):
    location_id, city = ctx.city()
    if ctx.rng.random() < 0.5:
        period, n = last_days(ctx)
        cond = f"date > {day_expr(ctx.today, -n)}"
    else:
        period, since = since_date(ctx)
        cond = f"date >= {since}"
    kind = ctx.rng.choice(["rain", "rain", "rain_days", "hottest_day", "max_temp"])
    if kind == "rain":
        q = ctx.pick([f"¿Cuánto ha llovido en {city} {period}?", f"Lluvia acumulada en {city} {period}"], [f"How much has it rained in {city} {period}?"])
        sql = f"SELECT sum(precipitation_sum_mm) FROM daily_summary WHERE location_id = '{location_id}' AND {cond}"
    elif kind == "rain_days":
        q = ctx.pick([f"¿Cuántos días ha llovido en {city} {period}?", f"Días de lluvia en {city} {period}"], [f"How many rainy days in {city} {period}?"])
        sql = f"SELECT count(*) FROM daily_summary WHERE location_id = '{location_id}' AND {cond} AND precipitation_sum_mm > 0"
    elif kind == "hottest_day":
        q = ctx.pick([f"¿Qué día hizo más calor en {city} {period}?", f"Día más caluroso en {city} {period}"], [f"Which was the hottest day in {city} {period}?"])
        sql = f"SELECT date, temperature_max_c FROM daily_summary WHERE location_id = '{location_id}' AND {cond} ORDER BY temperature_max_c DESC, date LIMIT 1"
    else:
        q = ctx.pick([f"¿Cuál ha sido la temperatura más alta en {city} {period}?", f"Máxima absoluta en {city} {period}"], [f"Highest temperature in {city} {period}"])
        sql = f"SELECT max(temperature_max_c) FROM daily_summary WHERE location_id = '{location_id}' AND {cond}"
    return q, sql


def ds_evolution(ctx: Ctx):
    location_id, city = ctx.city()
    period, n = last_days(ctx)
    column, es, en = ctx.rng.choice(OBSERVED_STATS)
    q = ctx.pick([f"Evolución de {es} en {city} {period}", f"{es.capitalize()} día a día en {city} {period}"],
                 [f"Daily {en} in {city} {period}"])
    return q, f"SELECT date, {column} FROM daily_summary WHERE location_id = '{location_id}' AND date > {day_expr(ctx.today, -n)} ORDER BY date"


def ds_national(ctx: Ctx):
    day, day_sql = observed_day(ctx)
    expr, es, en = ctx.rng.choice([
        ("avg(temperature_mean_c)", ["¿Cuál fue la temperatura media de todas las capitales {d}?", "Temperatura media nacional {d}"], ["Average temperature across all capitals {d}"]),
        ("sum(precipitation_sum_mm)", ["¿Cuánto llovió en total en todas las capitales {d}?", "Lluvia total sumando todas las capitales {d}"], ["Total rain over all capitals {d}"]),
        ("max(temperature_max_c)", ["¿Cuál fue la temperatura más alta de España {d}?", "Máxima más alta entre las capitales {d}"], ["Highest temperature in any capital {d}"]),
        ("min(temperature_min_c)", ["¿Cuál fue la temperatura más baja de España {d}?", "Mínima más baja entre las capitales {d}"], ["Lowest temperature in any capital {d}"]),
        ("count(*) FILTER (WHERE precipitation_sum_mm > 0)", ["¿En cuántas capitales llovió {d}?", "Número de capitales con lluvia {d}"], ["In how many capitals did it rain {d}?"]),
    ])
    return ctx.pick(es, en).format(d=day), f"SELECT {expr} FROM daily_summary WHERE date = {day_sql}"


MAE_BANDS = [  # column, Spanish, English
    ("temperature_mae_lead_1_6h", "con 1 a 6 horas de antelación", "1 to 6 hours ahead"),
    ("temperature_mae_lead_7_24h", "con 7 a 24 horas de antelación", "7 to 24 hours ahead"),
    ("temperature_mae_lead_25_48h", "con 25 a 48 horas de antelación", "25 to 48 hours ahead"),
    ("temperature_mae_lead_49_72h", "con 49 a 72 horas de antelación", "49 to 72 hours ahead"),
]
ALL_BANDS = ", ".join(b[0] for b in MAE_BANDS)


def ds_skill_city(ctx: Ctx):
    location_id, city = ctx.city()
    day, day_sql = observed_day(ctx)
    if ctx.rng.random() < 0.5:
        q = ctx.pick([f"¿Cuánto acertó la previsión de temperatura en {city} {day}?", f"Error de la previsión de temperatura en {city} {day} por plazo"],
                     [f"How accurate was the temperature forecast in {city} {day}?"])
        cols = ALL_BANDS
    else:
        column, es, en = ctx.rng.choice(MAE_BANDS)
        q = ctx.pick([f"Error medio de la previsión de temperatura {es} en {city} {day}", f"¿Cuánto se equivocó la previsión {es} en {city} {day}?"],
                     [f"Temperature forecast error {en} in {city} {day}"])
        cols = column
    return q, f"SELECT {cols} FROM daily_summary WHERE location_id = '{location_id}' AND date = {day_sql}"


def ds_skill_ranking(ctx: Ctx):
    day, day_sql = observed_day(ctx)
    n = ctx.rng.choice([1, 3, 5])
    column, es, en = ctx.rng.choice(MAE_BANDS)
    best = ctx.rng.random() < 0.5
    word_es, word_en, order = ("más precisa", "most accurate", "ASC") if best else ("menos precisa", "least accurate", "DESC")
    q = ctx.pick([f"¿En qué {n} capitales fue la previsión {word_es} {es} {day}?" if n > 1 else f"¿En qué capital fue la previsión {word_es} {es} {day}?"],
                 [f"Where was the forecast {word_en} {en} {day}? Top {n}"])
    sql = f"SELECT location_id, {column} FROM daily_summary WHERE date = {day_sql} AND {column} IS NOT NULL ORDER BY {column} {order}, location_id LIMIT {n}"
    return q, sql


def ds_skill_national(ctx: Ctx):
    day, day_sql = observed_day(ctx)
    q = ctx.pick([f"Error medio de la previsión de temperatura en todas las capitales {day}, por plazo", f"¿Cómo empeora la previsión con la antelación {day}?"],
                 [f"Average temperature forecast error over all capitals {day}, by lead time"])
    cols = ", ".join(f"avg({b[0]})" for b in MAE_BANDS)
    return q, f"SELECT {cols} FROM daily_summary WHERE date = {day_sql}"


# ================================================================ forecast_vs_observed

FVO_VARS = [  # error column, Spanish, English
    ("temperature_c", "temperatura", "temperature"),
    ("relative_humidity_pct", "humedad", "humidity"),
    ("pressure_msl_hpa", "presión", "pressure"),
    ("wind_speed_kmh", "viento", "wind speed"),
]


def fvo_var(ctx: Ctx):
    return FVO_VARS[0] if ctx.rng.random() < 0.5 else ctx.rng.choice(FVO_VARS[1:])


def fvo_mae_by_lead(ctx: Ctx):
    location_id, city = ctx.city()
    day, day_sql = observed_day(ctx)
    column, es, en = fvo_var(ctx)
    q = ctx.pick([f"Error absoluto medio de la previsión de {es} por horas de antelación en {city} {day}", f"¿Cómo varía el error de {es} con el plazo de previsión en {city} {day}?"],
                 [f"Mean absolute {en} forecast error by lead time in {city} {day}"])
    sql = (f"SELECT lead_hours, avg(abs(error_{column})) AS mae FROM forecast_vs_observed "
           f"WHERE location_id = '{location_id}' AND date = {day_sql} GROUP BY lead_hours ORDER BY lead_hours")
    return q, sql


def fvo_bias(ctx: Ctx):
    location_id, city = ctx.city()
    column, es, en = fvo_var(ctx)
    lead = ctx.rng.choice([None, None, 1, 6, 12, 24, 48])
    lead_es = f" a {lead} horas vista" if lead else ""
    lead_en = f" {lead} hours ahead" if lead else ""
    func = ctx.rng.choice(["avg", "mae"])
    if func == "avg":
        q = ctx.pick([f"¿La previsión sobrestima o subestima la {es} en {city}{lead_es}?", f"Sesgo de la previsión de {es} en {city}{lead_es}"],
                     [f"Does the forecast over- or underestimate {en} in {city}{lead_en}?", f"{en.capitalize()} forecast bias in {city}{lead_en}"])
        expr = f"avg(error_{column})"
    else:
        q = ctx.pick([f"Error absoluto medio de la previsión de {es} en {city}{lead_es}", f"¿Cuánto falla de media la previsión de {es} en {city}{lead_es}?"],
                     [f"Mean absolute {en} forecast error in {city}{lead_en}"])
        expr = f"avg(abs(error_{column}))"
    cond = f" AND lead_hours = {lead}" if lead else ""
    return q, f"SELECT {expr} FROM forecast_vs_observed WHERE location_id = '{location_id}'{cond}"


def fvo_at_time(ctx: Ctx):
    location_id, city = ctx.city()
    day, day_sql = observed_day(ctx)
    hour = ctx.rng.randint(0, 23)
    lead = ctx.rng.choice([1, 3, 6, 12, 24, 24, 48])
    q = ctx.pick([f"¿Qué temperatura se pronosticó con {lead} horas de antelación para {city} {day} a las {hour}:00 y cuál se observó?",
                  f"Temperatura prevista a {lead} h y observada en {city} {day} a las {hour}:00"],
                 [f"Forecast ({lead} h ahead) vs observed temperature in {city} {day} at {hour}:00"])
    sql = (f"SELECT forecast_temperature_c, observed_temperature_c FROM forecast_vs_observed "
           f"WHERE location_id = '{location_id}' AND valid_at = {hour_expr(day_sql, hour)} AND lead_hours = {lead}")
    return q, sql


def fvo_max_error(ctx: Ctx):
    location_id, city = ctx.city()
    day, day_sql = observed_day(ctx)
    column, es, en = fvo_var(ctx)
    q = ctx.pick([f"¿Cuál fue el mayor error de la previsión de {es} en {city} {day}?", f"Fallo máximo de la previsión de {es} en {city} {day}"],
                 [f"Largest {en} forecast error in {city} {day}"])
    return q, f"SELECT max(abs(error_{column})) FROM forecast_vs_observed WHERE location_id = '{location_id}' AND date = {day_sql}"


def fvo_ranking(ctx: Ctx):
    period, n_days = last_days(ctx)
    n = ctx.rng.choice([3, 5, 10])
    column, es, en = fvo_var(ctx)
    worst = ctx.rng.random() < 0.6
    word_es, word_en, order = ("mayor", "largest", "DESC") if worst else ("menor", "smallest", "ASC")
    q = ctx.pick([f"Las {n} capitales con {word_es} error de previsión de {es} {period}", f"¿Dónde ha tenido {word_es} error la previsión de {es} {period}? Top {n}"],
                 [f"The {n} capitals with the {word_en} {en} forecast error {period}"])
    sql = (f"SELECT location_id, avg(abs(error_{column})) AS mae FROM forecast_vs_observed WHERE date > {day_expr(ctx.today, -n_days)} "
           f"GROUP BY location_id ORDER BY mae {order}, location_id LIMIT {n}")
    return q, sql


# ---------------------------------------------------------------- registry

FAMILIES: dict[str, tuple[str, Callable[[Ctx], tuple[str, str]], int]] = {
    # name: (table that sets "today", function, examples to try)
    "fc_value_at_hour": ("forecast", fc_value_at_hour, 260),
    "fc_stat_day": ("forecast", fc_stat_day, 260),
    "fc_rain_day": ("forecast", fc_rain_day, 160),
    "fc_rain_next_hours": ("forecast", fc_rain_next_hours, 120),
    "fc_ranking": ("forecast", fc_ranking, 220),
    "fc_threshold": ("forecast", fc_threshold, 200),
    "fc_hourly": ("forecast", fc_hourly, 140),
    "fc_when_peak": ("forecast", fc_when_peak, 160),
    "fc_compare": ("forecast", fc_compare, 160),
    "fc_daily_table": ("forecast", fc_daily_table, 100),
    "fc_issued": ("forecast", fc_issued, 12),
    "st_station_of": ("static", st_station_of, 80),
    "st_attribute": ("static", st_attribute, 140),
    "st_ranking": ("static", st_ranking, 60),
    "st_threshold": ("static", st_threshold, 30),
    "ds_value": ("observed", ds_value, 260),
    "ds_all_stats": ("observed", ds_all_stats, 100),
    "ds_ranking": ("observed", ds_ranking, 200),
    "ds_period_total": ("observed", ds_period_total, 200),
    "ds_evolution": ("observed", ds_evolution, 100),
    "ds_national": ("observed", ds_national, 120),
    "ds_skill_city": ("observed", ds_skill_city, 120),
    "ds_skill_ranking": ("observed", ds_skill_ranking, 100),
    "ds_skill_national": ("observed", ds_skill_national, 40),
    "fvo_mae_by_lead": ("observed", fvo_mae_by_lead, 100),
    "fvo_bias": ("observed", fvo_bias, 140),
    "fvo_at_time": ("observed", fvo_at_time, 140),
    "fvo_max_error": ("observed", fvo_max_error, 80),
    "fvo_ranking": ("observed", fvo_ranking, 100),
}


def real_today_for(kind: str, rng: random.Random) -> date:
    """The real snapshot date an example is anchored to (before shifting)."""
    if kind == "observed":
        # any day with at least one earlier day of data, so "ayer" exists
        return FIRST_OBSERVED + timedelta(days=rng.randint(1, (LAST_OBSERVED - FIRST_OBSERVED).days))
    return FORECAST_DAY
