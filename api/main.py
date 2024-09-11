import os

from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
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
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)


if ENV == "dev":
    host = HOST
else:
    host = f"/cloudsql/{INSTANCE_CONNECTION_NAME}"

pool = ThreadedConnectionPool(
    1, 10, user=USERNAME, password=PASSWORD, host=HOST, database=DATABASE
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


@app.get("/demographics")
async def read_demographics(
    category: Annotated[str, Query(max_length=100)], db=Depends(get_db)
):
    with db.cursor() as cur:
        cur.execute(
            """
            select tract_id, "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"
            from dev.api__demographics where description = %s
            """,
            (category,),
        )
        return [[desc[0] for desc in cur.description]] + cur.fetchall()


@app.get("/census-tracts")
async def read_census_tracts(year: Year, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute("select * from dev.api__census_tracts where year_ = %s", (year,))
        row = cur.fetchone()

    return row[1] if row is not None else None


@app.get("/high-frequency-transit-lines")
async def read_high_frequency_transit_lines(year: Year, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute(
            """
            select line_geom_json
            from dev.api__high_frequency_transit_lines
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
            from dev.api__high_frequency_transit_lines
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
            from dev.api__high_frequency_transit_lines
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
            from dev.api__high_frequency_transit_lines
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
    samples: Annotated[int, Query(ge=0, le=1000)] = 100,
    db=Depends(get_db),
    predictor=Depends(get_predictor),
):
    result = predictor.predict(
        db,
        samples=samples,
        intervention=(
            {
                "radius_blue": blue_zone_radius,
                "limit_blue": blue_zone_limit,
                "radius_yellow": yellow_zone_line_radius,
                "limit_yellow": yellow_zone_limit,
            }
        ),
    )
    return (
        [str(x) for x in result["census_tracts"].tolist()],
        result["housing_units"].tolist(),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
