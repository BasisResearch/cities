#!/usr/bin/env python

import psycopg2

PARCEL_YEARS = range(2002, 2018)
COUNTY_ID = "053"

conn = psycopg2.connect(database="cities")
cur = conn.cursor()


with open("etl/schema.sql", "r") as f:
    cur.execute(f.read())
conn.commit()

# select distinct geometry from all parcel tables
distinct_geom = " union ".join(
    f"select geom from parcel_raw_{year} where city = 'MINNEAPOLIS'"
    for year in PARCEL_YEARS
)
parcel_geom_load = f"insert into parcel_geom (parcel_geom_data) {distinct_geom};"
print("Executing:", parcel_geom_load)
cur.execute(parcel_geom_load)
conn.commit()

# insert parcel data into parcel table
parcel_data = " union all ".join(
    f"""
    select replace(pin, '{COUNTY_ID}-', ''), {year}, emv_land, emv_bldg, emv_total, nullif(year_built, 0), sale_date, sale_value, parcel_geom_id
    from parcel_raw_{year}, parcel_geom
    where parcel_raw_{year}.geom = parcel_geom.parcel_geom_data
      and city = 'MINNEAPOLIS'
    """
    for year in PARCEL_YEARS
)
parcel_load = f"""
insert into parcel (parcel_id, parcel_year, parcel_emv_land, parcel_emv_building, parcel_emv_total, parcel_year_built, parcel_sale_date, parcel_sale_value, parcel_geom_id)
    {parcel_data}
    """
print("Executing:", parcel_load)
cur.execute(parcel_load)
conn.commit()
