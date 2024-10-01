import os

from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Query
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

load_dotenv()

ENV = os.getenv("ENV")
USERNAME = os.getenv("DB_USERNAME")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")
DB_SEARCH_PATH = os.getenv("DB_SEARCH_PATH")
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")

app = FastAPI()

if ENV == "dev":
    from fastapi.middleware.cors import CORSMiddleware

    origins = [
        "http://localhost",
        "http://localhost:5000",
    ]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True)

app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)


if ENV == "dev":
    host = HOST
else:
    host = f"/cloudsql/{INSTANCE_CONNECTION_NAME}"

pool = ThreadedConnectionPool(
    1,
    10,
    user=USERNAME,
    password=PASSWORD,
    host=HOST,
    database=DATABASE,
    options=f"-csearch_path={DB_SEARCH_PATH}",
)


def get_db() -> psycopg2.extensions.connection:
    db = pool.getconn()
    try:
        yield db
    finally:
        pool.putconn(db)


predictor = None


def get_predictor(db: psycopg2.extensions.connection = Depends(get_db)):
    from cities.deployment.tracts_minneapolis.predict import TractsModelPredictor

    global predictor
    if predictor is None:
        predictor = TractsModelPredictor(db)
    return predictor


Limit = Annotated[float, Query(ge=0, le=1)]
Radius = Annotated[float, Query(ge=0)]
Year = Annotated[int, Query(ge=2000, le=2030)]


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "public, max-age=300"
    return response


if ENV == "dev":

    @app.middleware("http")
    async def add_acess_control_header(request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response


@app.get("/demographics")
async def read_demographics(
    category: Annotated[str, Query(max_length=100)], db=Depends(get_db)
):
    with db.cursor() as cur:
        cur.execute(
            """
            select tract_id, "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"
            from api__demographics where description = %s
            """,
            (category,),
        )
        return [[desc[0] for desc in cur.description]] + cur.fetchall()


@app.get("/census-tracts")
async def read_census_tracts(year: Year, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute("select * from api__census_tracts where year_ = %s", (year,))
        row = cur.fetchone()

    return row[1] if row is not None else None


@app.get("/high-frequency-transit-lines")
async def read_high_frequency_transit_lines(year: Year, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute(
            """
            select line_geom_json
            from api__high_frequency_transit_lines
            where '%s-01-01'::date <@ valid
            """,
            (year,),
        )
        row = cur.fetchone()

    return row[0] if row is not None else None


@app.get("/high-frequency-transit-stops")
async def read_high_frequency_transit_stops(year: Year, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute(
            """
            select stop_geom_json
            from api__high_frequency_transit_lines
            where '%s-01-01'::date <@ valid
            """,
            (year,),
        )
        row = cur.fetchone()

    return row[0] if row is not None else None


@app.get("/yellow-zone")
async def read_yellow_zone(
    year: Year, line_radius: Radius, stop_radius: Radius, db=Depends(get_db)
):
    with db.cursor() as cur:
        cur.execute(
            """
            select
              st_asgeojson(st_transform(st_union(st_buffer(line_geom, %s, 'quad_segs=4'), st_buffer(stop_geom, %s, 'quad_segs=4')), 4269))::json
            from api__high_frequency_transit_lines
            where '%s-01-01'::date <@ valid
            """,
            (line_radius, stop_radius, year),
        )
        row = cur.fetchone()

    if row is None:
        return None

    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"id": "0"}, "geometry": row[0]}
        ],
    }


@app.get("/blue-zone")
async def read_blue_zone(year: Year, radius: Radius, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute(
            """
            select st_asgeojson(st_transform(st_buffer(line_geom, %s, 'quad_segs=4'), 4269))::json
            from api__high_frequency_transit_lines
            where '%s-01-01'::date <@ valid
            """,
            (radius, year),
        )
        row = cur.fetchone()

    if row is None:
        return None

    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"id": "0"}, "geometry": row[0]}
        ],
    }


@app.get("/predict")
async def read_predict(
    blue_zone_radius: Radius,
    yellow_zone_line_radius: Radius,
    yellow_zone_stop_radius: Radius,
    blue_zone_limit: Limit,
    yellow_zone_limit: Limit,
    year: Year,
    db=Depends(get_db),
    predictor=Depends(get_predictor),
):
    result = predictor.predict_cumulative(
        db,
        intervention=(
            {
                "radius_blue": blue_zone_radius,
                "limit_blue": blue_zone_limit,
                "radius_yellow_line": yellow_zone_line_radius,
                "radius_yellow_stop": yellow_zone_stop_radius,
                "limit_yellow": yellow_zone_limit,
                "reform_year": year,
            }
        ),
    )
    return {
        "census_tracts": [str(t) for t in result["census_tracts"]],
        "housing_units_factual": [t.item() for t in result["housing_units_factual"]],
        "housing_units_counterfactual": [
            t.tolist() for t in result["housing_units_counterfactual"]
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
