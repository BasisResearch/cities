import psycopg2

from db import HOST, USER

PARCEL_YEARS = range(2002, 2024)
COUNTY_ID = "053"

conn = psycopg2.connect(host=HOST, user=USER, database="cities")
cur = conn.cursor()

with open("etl/zip_schema.sql", "r") as f:
    cur.execute(f.read())
conn.commit()

zip_load = """
insert into zip_code(zip_code, year, geom)
select zcta5ce20, 2020, geom from zip_raw_2020
union select zcta, 2000, ST_Transform(geom, 4269) from zip_raw_2000
"""
print("Executing:", zip_load)
cur.execute(zip_load)
conn.commit()
