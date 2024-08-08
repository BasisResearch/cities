#!/usr/bin/env python

import logging
import os
import glob
from io import StringIO

import psycopg2
import pandas as pd

from db import HOST, USER

log = logging.getLogger(__name__)

RAW_DATA_DIRECTORY = "zoning/data/raw/demographics/zip_codes/fair_market_rents"


def preprocess_csv_to_df(filename):
    year = filename.split("_")[-1].replace(".csv", "")

    df = pd.read_csv(filename, dtype=str, na_values={})

    rename_dict = {}
    for col in list(df.columns):
        if "zip" in col.lower() or col == "zcta":
            rename_dict[col] = "zip_code"
        elif "BR" in col and "90" not in col and "110" not in col:
            rename_dict[col] = "rent_br" + col.lower().split("br")[0][-1]
        elif "area_rent_br" in col:
            rename_dict[col] = "rent_br" + col[-1]
        elif "safmr" in col and "90" not in col and "110" not in col:
            rename_dict[col] = "rent_br" + col.split("_")[-1][0]

    df = df.rename(columns=rename_dict)[
        [
            "zip_code",
            "rent_br0",
            "rent_br1",
            "rent_br2",
            "rent_br3",
            "rent_br4",
        ]
    ]

    for col in df.columns:
        if "rent_" in col:
            df[col] = [x.replace("$", "").replace(",", "") for x in df[col]]

    return (year, df)


def copy_from_stringio(cur, df, table):
    """Here we are going save the dataframe in memory and use copy_from() to copy it to the table"""
    buf = StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cur.copy_from(buf, table, sep=",")


def main():
    conn = psycopg2.connect(host=HOST, user=USER, database="cities")
    cur = conn.cursor()

    with open("etl/fair_market_rents_schema.sql", "r") as f:
        cur.execute(f.read())

    cur.execute("drop table if exists fmr_temp")
    cur.execute(
        """
        create temp table fmr_temp (
        zip text
        , rent_br0 numeric
        , rent_br1 numeric
        , rent_br2 numeric
        , rent_br3 numeric
        , rent_br4 numeric)
        """
    )

    for filename in glob.glob(f"{RAW_DATA_DIRECTORY}/*.csv"):
        (year, df) = preprocess_csv_to_df(filename)
        cur.execute("truncate fmr_temp")
        copy_from_stringio(cur, df, "fmr_temp")

        cur.execute(
            "insert into fair_market_rents_raw select *, %s as year from fmr_temp",
            (year,),
        )
    conn.commit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
