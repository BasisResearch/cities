#!/usr/bin/env python

import glob
import subprocess
import logging
import os

from db import HOST, USER

log = logging.getLogger(__name__)

BASE_DIR = "zoning/data/raw"
OGR2OGR_OPTS = [
    "--config",
    "PG_USE_COPY",  # use postgres specific copy
    "-progress",
    "-lco",
    "PRECISION=NO",  # disable use of numeric types (required when shapefiles mis-specify numeric precision)
    "-overwrite",  # overwrite existing tables
    "-lco",
    "GEOMETRY_NAME=geom",  # name of geometry column
    "-nlt",
    "PROMOTE_TO_MULTI",  # promote all POLYGONs to MULTIPOLYGONs
]
DB_OPTS = [f"Pg:dbname=cities host={HOST} user={USER} port=5432"]

# (shapefile, table_name) pairs. shapefiles are relative to BASE_DIR
REL_SHAPES = [
    (
        "base/shp_society_census2000tiger_zcta/Census2000TigerZipCodeTabAreas.shp",
        "zip_raw_2000",
    ),
    (
        "base/shp_bdry_zip_code_tabulation_areas/zip_code_tabulation_areas.shp",
        "zip_raw_2020",
    ),
    (
        "base/hennepin_county_census_tracts_2018/cb_2018_27_tract_500k.shp",
        "census_tract_raw_2018",
    ),
    (
        "base/hennepin_county_census_block_groups_2018/cb_2018_27_bg_500k.shp",
        "census_block_group_raw_2018",
    ),
    (
        "base/hennepin_county_census_tracts_2023/cb_2023_27_tract_500k.shp",
        "census_tract_raw_2023",
    ),
    (
        "base/hennepin_county_census_block_groups_2023/cb_2023_27_bg_500k.shp",
        "census_block_group_raw_2023",
    ),
    (
        "commercial_permits/shp_struc_non_res_construction/NonresidentialConstruction.shp",
        "commercial_permits_raw",
    ),
    (
        "residential_permits/shp_econ_residential_building_permts/ResidentialPermits.shp",
        "residential_permits_raw",
    ),
]


def main():
    # convert relative paths to absolute paths
    abs_shapes = [(os.path.join(BASE_DIR, shape), table) for shape, table in REL_SHAPES]

    for parcel_shape_dir in glob.glob(
        os.path.join(BASE_DIR, "property_values/shp_plan_regional_parcels_*/")
    ):
        year = int(parcel_shape_dir.split("/")[-2].split("_")[-1])
        shape = os.path.join(parcel_shape_dir, f"Parcels{year}Hennepin.shp")
        table = f"parcel_raw_{year}"
        abs_shapes.append((shape, table))

    log.info("Loading raw shape files: %s", abs_shapes)
    for shape, table in abs_shapes:
        if not os.path.exists(shape):
            log.warn("Skipping %s because it does not exist", shape)
            continue

        subprocess.check_call(
            ["ogr2ogr"] + OGR2OGR_OPTS + ["-nln", table] + DB_OPTS + [shape]
        )


if __name__ == "__main__":
    main()
