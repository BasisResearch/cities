#!/usr/bin/env python

import glob
import logging
import psycopg2

from db import HOST, USER

log = logging.getLogger(__name__)


RAW_DATA_DIRECTORY = "zoning/data/raw/demographics/zip_codes/usps_migration"


def main():
    conn = psycopg2.connect(host=HOST, user=USER, database="cities")
    cur = conn.cursor()

    with open("etl/usps_migration_raw_schema.sql", "r") as f:
        cur.execute(f.read())

    cur.execute("drop table if exists m_temp")
    cur.execute(
        """
        create temp table m_temp (
    yyyymm text
    , zip_code text
    , city text
    , state text
    , total_from_zip numeric
    , total_from_zip_business numeric
    , total_from_zip_family numeric
    , total_from_zip_individual numeric
    , total_from_zip_perm numeric
    , total_from_zip_temp numeric
    , total_to_zip numeric
    , total_to_zip_business numeric
    , total_to_zip_family numeric
    , total_to_zip_individual numeric
    , total_to_zip_perm numeric
    , total_to_zip_temp numeric
        )
        """
    )

    for filename in glob.glob(f"{RAW_DATA_DIRECTORY}/*.csv"):
        log.info(f"Loading {filename}")
        year = filename.split("/")[-1].split(".")[0].replace("Y", "")

        cur.execute("truncate m_temp")

        with open(filename, "r") as f:
            cur.copy_expert("copy m_temp from stdin with csv header", f)

        cur.execute(
            "insert into usps_migration_raw select *, %s from m_temp",
            (year,),
        )
    conn.commit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
