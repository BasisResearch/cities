import logging
import os
import tempfile

from dotenv import load_dotenv
from google.cloud import storage
import psycopg2
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# DATA INFO
PROJECT_NAME = os.getenv("GOOGLE_CLOUD_PROJECT")
BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET")

# DATABASE INFO
SCHEMA = os.getenv("SCHEMA")
HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

YEAR_RANGE = range(2013, 2023)
ACS_CODES = {
    "B03002_003E": "population_white_non_hispanic",
    "B03002_004E": "population_black_non_hispanic",
    "B03002_005E": "population_asian_non_hispanic",
    "B03002_006E": "population_native_hawaiian_or_pacific_islander_non_hispanic",
    "B03002_007E": "population_american_indian_or_alaska_native_non_hispanic",
    "B03002_008E": "population_other_non_hispanic",
    "B03002_009E": "population_multiple_races_non_hispanic",
    "B03002_010E": "population_multiple_races_and_other_non_hispanic",
    "B07204_001E": "geographic_mobility_total_responses",
    "B07204_002E": "geographic_mobility_same_house_1_year_ago",
    "B07204_004E": "geographic_mobility_different_house_1_year_ago_same_city",
    "B07204_005E": "geographic_mobility_different_house_1_year_ago_same_county",
    "B07204_006E": "geographic_mobility_different_house_1_year_ago_same_state",
    "B07204_007E": "geographic_mobility_different_house_1_year_ago_same_country",
    "B07204_016E": "geographic_mobility_different_house_1_year_ago_abroad",
    "B01003_001E": "population",
    "B02001_002E": "white",
    "B02001_003E": "black",
    "B02001_004E": "american_indian_or_alaska_native",
    "B02001_005E": "asian",
    "B02001_006E": "native_hawaiian_or_pacific_islander",
    "B03001_003E": "hispanic_or_latino",
    "B02001_007E": "other_race",
    "B02001_008E": "multiple_races",
    "B02001_009E": "multiple_races_and_other_race",
    "B02001_010E": "two_or_more_races_excluding_other",
    "B02015_002E": "east_asian_chinese",
    "B02015_003E": "east_asian_hmong",
    "B02015_004E": "east_asian_japanese",
    "B02015_005E": "east_asian_korean",
    "B02015_006E": "east_asian_mongolian",
    "B02015_007E": "east_asian_okinawan",
    "B02015_008E": "east_asian_taiwanese",
    "B02015_009E": "east_asian_other",
    "B02015_010E": "southeast_asian_burmese",
    "B02015_011E": "southeast_asian_cambodian",
    "B02015_012E": "southeast_asian_filipino",
    "B02015_013E": "southeast_asian_indonesian",
    "B02015_014E": "southeast_asian_laotian",
    "B02015_015E": "southeast_asian_malaysian",
    "B02015_016E": "southeast_asian_mien",
    "B02015_017E": "southeast_asian_singaporean",
    "B02015_018E": "southeast_asian_thai",
    "B02015_019E": "southeast_asian_viet",
    "B02015_020E": "southeast_asian_other",
    "B02015_021E": "south_asian_asian_indian",
    "B02015_022E": "south_asian_bangladeshi",
    "B02015_023E": "south_asian_bhutanese",
    "B02015_024E": "south_asian_nepalese",
    "B02015_025E": "south_asian_pakistani",
    "B19013_001E": "median_household_income",
    "B19013A_001E": "median_household_income_white",
    "B19013H_001E": "median_household_income_white_non_hispanic",
    "B19013I_001E": "median_household_income_hispanic",
    "B19013B_001E": "median_household_income_black",
    "B19013C_001E": "median_household_income_american_indian_or_alaska_native",
    "B19013D_001E": "median_household_income_asian",
    "B19013E_001E": "median_household_income_native_hawaiian_or_pacific_islander",
    "B19013F_001E": "median_household_income_other_race",
    "B19013G_001E": "median_household_income_multiple_races",
    "B19019_002E": "median_household_income_1_person_households",
    "B19019_003E": "median_household_income_2_person_households",
    "B19019_004E": "median_household_income_3_person_households",
    "B19019_005E": "median_household_income_4_person_households",
    "B19019_006E": "median_household_income_5_person_households",
    "B19019_007E": "median_household_income_6_person_households",
    "B19019_008E": "median_household_income_7_or_more_person_households",
    "B01002_001E": "median_age",
    "B01002_002E": "median_age_male",
    "B01002_003E": "median_age_female",
    "B25031_001E": "median_gross_rent",
    "B25031_002E": "median_gross_rent_0_bedrooms",
    "B25031_003E": "median_gross_rent_1_bedrooms",
    "B25031_004E": "median_gross_rent_2_bedrooms",
    "B25031_005E": "median_gross_rent_3_bedrooms",
    "B25031_006E": "median_gross_rent_4_bedrooms",
    "B25031_007E": "median_gross_rent_5_bedrooms",
    "B25032_001E": "total_housing_units",
    "B25032_002E": "total_owner_occupied_housing_units",
    "B25032_013E": "total_renter_occupied_housing_units",
    "B25070_001E": "median_gross_rent_as_percentage_of_household_income",
}


if __name__ == "__main__":
    conn = psycopg2.connect(
        host=HOST, database=DATABASE, user=USERNAME, password=PASSWORD
    )
    storage_client = storage.Client(project=PROJECT_NAME)
    bucket = storage_client.bucket(BUCKET_NAME)
    cur = conn.cursor()

    cur.execute(
        f"create table if not exists {SCHEMA}.acs_tract_raw (statefp text, countyfp text, tractce text, year int, code text, value numeric)"
    )
    cur.execute(f"truncate table {SCHEMA}.acs_tract_raw")

    temp_table = f"{SCHEMA}.acs_tract_temp"
    cur.execute(f"drop table if exists {temp_table}")
    cur.execute(
        f"create table {temp_table} (statefp text, countyfp text, tractce text, value numeric)"
    )
    for code in tqdm(ACS_CODES.keys()):
        desc = ACS_CODES[code]

        blobs = list(bucket.list_blobs(prefix=f"acs/tracts/{desc}/"))
        if len(blobs) == 0:
            logging.info(f"No blobs found for {desc}")
            continue

        for blob in blobs:
            year = blob.name.split("/")[-1].split(".")[0]
            cur.execute(f"truncate {temp_table}")
            with tempfile.NamedTemporaryFile() as temp:
                blob.download_to_filename(temp.name)
                cur.copy_expert(f"copy {temp_table} from stdin with csv header", temp)

            cur.execute(
                f"insert into {SCHEMA}.acs_tract_raw select statefp, countyfp, tractce, %s, %s, value from {temp_table}",
                (year, code),
            )
    cur.execute(f"drop table {temp_table}")
    conn.commit()

    cur.execute(
        f"create table if not exists {SCHEMA}.acs_bg_raw (statefp text, countyfp text, tractce text, blkgrpce text, year int, code text, value numeric)"
    )
    cur.execute(f"truncate table {SCHEMA}.acs_bg_raw")

    temp_table = f"{SCHEMA}.acs_tract_temp"
    cur.execute(f"drop table if exists {temp_table}")
    cur.execute(
        f"create table {temp_table} (statefp text, countyfp text, tractce text, blkgrpce text, value numeric)"
    )

    for code in tqdm(ACS_CODES.keys()):
        desc = ACS_CODES[code]

        blobs = list(bucket.list_blobs(prefix=f"acs/block_groups/{desc}/"))
        if len(blobs) == 0:
            logging.info(f"No blobs found for {desc}")
            continue

        for blob in blobs:
            year = blob.name.split("/")[-1].split(".")[0]
            cur.execute(f"truncate {temp_table}")
            with tempfile.NamedTemporaryFile() as temp:
                blob.download_to_filename(temp.name)
                cur.copy_expert(f"copy {temp_table} from stdin with csv header", temp)

            cur.execute(
                f"insert into {SCHEMA}.acs_bg_raw select statefp, countyfp, tractce, blkgrpce, %s, %s value from {temp_table}",
                (year, code),
            )
    cur.execute(f"drop table {temp_table}")
    conn.commit()
