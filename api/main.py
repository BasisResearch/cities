import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import uvicorn

from cities.deployment.tracts_minneapolis.predict import TractsModelPredictor

load_dotenv()

ENV = os.getenv("ENV")
USERNAME = os.getenv("USERNAME")
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


def get_db() -> psycopg2.extensions.connection:
    if ENV == "dev":
        db = psycopg2.connect(
            host=HOST, database=DATABASE, user=USERNAME, password=PASSWORD
        )
    else:
        db = psycopg2.connect(
            host=f"/cloudsql/{INSTANCE_CONNECTION_NAME}",
            database=DATABASE,
            user=USERNAME,
            password=PASSWORD,
        )

    try:
        yield db
    finally:
        db.close()


def get_predictor(
    db: psycopg2.extensions.connection = Depends(get_db),
) -> TractsModelPredictor:
    return TractsModelPredictor(db)


Limit = Annotated[float, Query(ge=0, le=1)]
Radius = Annotated[float, Query(ge=0)]
Year = Annotated[int, Query(ge=2000, le=2030)]


@app.get("/demographics")
async def read_demographics(
    category: Annotated[str, Query(max_length=100)], db=Depends(get_db)
):
    with db.cursor() as cur:
        cur.execute(
            "select * from dev.api__demographics where description = %s", (category,)
        )
        return cur.fetchall()


@app.get("/census-tracts")
async def read_census_tracts(year: Year, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute("select * from dev.api__census_tracts where year_ = %s", (year,))
        return cur.fetchone()


@app.get("/high-frequency-transit")
async def read_high_frequency_transit(year: Year, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute(
            """
            select line_geom_json, stop_geom_json
            from dev.api__high_frequency_transit_lines
            where '%s-01-01'::date <@ valid
            """,
            (year,),
        )
        line_geom, stop_geom = cur.fetchone()
    return {"line_geom": line_geom, "stop_geom": stop_geom}


@app.get("/yellow-zone")
async def read_yellow_zone(
    year: Year, line_radius: Radius, stop_radius: Radius, db=Depends(get_db)
):
    with db.cursor() as cur:
        cur.execute(
            """
            select
              st_asgeojson(st_union(st_buffer(line_geom, %s, 'quad_segs=4'),
                                    st_buffer(stop_geom, %s, 'quad_segs=4')))::json
            from dev.api__high_frequency_transit_lines
            where '%s-01-01'::date <@ valid
            """,
            (line_radius, stop_radius, year),
        )
        return cur.fetchone()[0]


@app.get("/blue-zone")
async def read_blue_zone(year: Year, radius: Radius, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute(
            """
            select st_asgeojson(st_buffer(line_geom, %s, 'quad_segs=4'))::json
            from dev.api__high_frequency_transit_lines
            where '%s-01-01'::date <@ valid
            """,
            (radius, year),
        )
        return cur.fetchone()[0]


@app.get("/predict")
async def read_predict(
    blue_zone_radius: Radius,
    yellow_zone_line_radius: Radius,
    yellow_zone_stop_radius: Radius,
    blue_zone_limit: Limit,
    yellow_zone_limit: Limit,
    year: Year,
    samples: Annotated[int, Query(ge=0, le=1000)] = 100,
    predictor: TractsModelPredictor = Depends(get_predictor),
):
    result = predictor.predict(
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
