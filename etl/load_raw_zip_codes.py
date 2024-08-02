#!/usr/bin/env python

import glob
import os

SRID = 26915  # UTM Zone 15N


os.system(
    f"""
    shp2pgsql -s {SRID} -I -d zoning/data/raw/base/shp_society_census2000tiger_zcta/Census2000TigerZipCodeTabAreas.shp zip_raw_2000 | pv -l | psql --quiet cities
    """,
)

os.system(
    f"""
    shp2pgsql -s {SRID} -I -d zoning/data/raw/base/shp_bdry_zip_code_tabulation_areas/zip_code_tabulation_areas.shp zip_raw_2020 | pv -l | psql --quiet cities
    """,
)
