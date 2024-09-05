import os

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

import psycopg2
from typing import Dict

from cities.deployment.tracts_minneapolis.predict import TractsModelPredictor

USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")

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
    db = psycopg2.connect(
        host=HOST, database=DATABASE, user=USERNAME, password=PASSWORD
    )
    try:
        yield db
    finally:
        db.close()


def get_predictor(
    db: psycopg2.extensions.connection = Depends(get_db),
) -> TractsModelPredictor:
    return TractsModelPredictor(db)


@app.get("/demographics")
async def read_demographics(category: str, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("select * from api__demographics where description = %s", (category,))
    return cur.fetchall()


@app.get("/census_tracts")
async def read_census_tracts(year: int, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute(
        """
        with census_tracts as (
          select census_tract, geom from api__census_tracts
          where year_ = %s
        )
        select json_build_object('type', 'FeatureCollection', 'features', json_agg(ST_AsGeoJSON(census_tracts.*)::json))
        from census_tracts
        """,
        (year,),
    )
    return cur.fetchall()


@app.get("/predict")
async def read_predict(
    samples=100,
    intervention: Dict[str, float] = None,
    predictor: TractsModelPredictor = Depends(get_predictor),
):
    result = predictor.predict(samples=samples, intervention=intervention)
    result = [
        {
            "description": "predict",
            "tract_id": str(result["census_tracts"][i].item()),
            "2022": result["housing_units"][i].item(),
        }
        for i in range(len(result["census_tracts"]))
    ]
    print(result)
    return result
