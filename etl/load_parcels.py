#!/usr/bin/env python

import psycopg2

from db import HOST, USER

PARCEL_YEARS = range(2002, 2024)
COUNTY_ID = "053"

conn = psycopg2.connect(host=HOST, user=USER, database="cities")
cur = conn.cursor()

with open("etl/schema.sql", "r") as f:
    cur.execute(f.read())
conn.commit()

# select distinct geometry from all parcel tables
distinct_geom = " union ".join(
    f"select geom from parcel_raw_{year} where upper({'city' if year < 2018 else 'ctu_name'}) = 'MINNEAPOLIS'"
    for year in PARCEL_YEARS
)
parcel_geom_load = f"insert into parcel_geom (geom) {distinct_geom};"
print("Executing:", parcel_geom_load)
cur.execute(parcel_geom_load)
conn.commit()

# insert parcel data into parcel table
parcel_data = " union all ".join(
    f"""
    select replace(pin, '{COUNTY_ID}-', ''), '[{year}-01-01,{year+1}-01-01)'::daterange, nullif(emv_land, 0), nullif(emv_bldg, 0), nullif(emv_total, 0), nullif(year_built, 0), sale_date, nullif(sale_value, 0), parcel_geom.id
    from parcel_raw_{year}, parcel_geom
    where parcel_raw_{year}.geom = parcel_geom.geom
      and upper({'city' if year < 2018 else 'ctu_name'}) = 'MINNEAPOLIS'
    """
    for year in range(2002, 2018)
)

parcel_load = f"""
insert into parcel (pid, valid, emv_land, emv_building, emv_total, year_built, sale_date, sale_value, geom_id)
    {parcel_data}
    """
print("Executing:", parcel_load)
cur.execute(parcel_load)
conn.commit()
