#!/usr/bin/env python

import glob
import os

SRID = 26915  # UTM Zone 15N

for parcel_shape_dir in glob.glob(
    "zoning/data/raw/property_values/shp_plan_regional_parcels_*/"
):
    year = int(parcel_shape_dir.split("/")[-2].split("_")[-1])
    print(f"Loading parcels for year {year} from {parcel_shape_dir}")

    os.system(
        f"""
        shp2pgsql -s {SRID} -I -d {parcel_shape_dir}Parcels{year}Hennepin.shp parcel_raw_{year} | pv -l | psql --quiet cities
        """,
    )
