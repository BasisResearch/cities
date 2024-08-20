import os
import re
from dotenv import load_dotenv
import subprocess
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time
import logging
from google.cloud import storage
import argparse
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

# Paths inside the bucket
FOLDERS = [
    "fair_market_rents",
]

# DATABASE INFO
SCHEMA = os.getenv("SCHEMA")
HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

OGR2OGR_OPTS = [
    "--config",
    "PG_USE_COPY",
    "YES",
    "-progress",
    "-lco",
    "PRECISION=NO",
    "-overwrite",
    "-lco",
    "GEOMETRY_NAME=geom",
    "-nlt",
    "PROMOTE_TO_MULTI",
]
DB_OPTS = [
    f"PG:dbname={DATABASE} host={HOST} user={USERNAME} password={PASSWORD} port=5432"
]

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def get_db_connection():
    """Create a database connection with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            conn = psycopg2.connect(
                host=HOST, database=DATABASE, user=USERNAME, password=PASSWORD
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            return conn
        except psycopg2.OperationalError as e:
            if attempt < MAX_RETRIES - 1:
                logging.warning(
                    f"Connection attempt {attempt + 1} failed. Retrying in {RETRY_DELAY} seconds..."
                )
                time.sleep(RETRY_DELAY)
            else:
                logging.error(
                    f"Failed to connect to the database after {MAX_RETRIES} attempts: {e}"
                )
                raise


def create_schema_if_not_exists(conn):
    """Create the schema if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute(f"create schema if not exists {SCHEMA};")
        cur.execute("create extension if not exists postgis;")


def generate_table_name(blob_name):
    """Generate a PostgreSQL-friendly table name from the blob name, including all parent folders and removing duplicates."""
    table_name = os.path.splitext(blob_name)[0]
    path_components = table_name.split("/")

    # Remove any leading empty components
    path_components = [comp for comp in path_components if comp]

    table_name = "_".join(path_components)
    table_name = table_name.replace("-", "_").replace(".", "_")

    words = table_name.split("_")
    unique_words = []
    for word in words:
        if word.lower() not in (w.lower() for w in unique_words):
            unique_words.append(word)

    table_name = "_".join(unique_words)
    table_name = re.sub("_+", "_", table_name)

    if table_name[0].isdigit():
        table_name = "f_" + table_name

    if len(table_name) > 63:
        table_name = table_name[:63]

    table_name = table_name.rstrip("_")

    return table_name.lower()


def drop_table_if_exists(conn, table_name):
    """Drop the table if it exists."""
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{table_name} CASCADE;")


def load_into_server(conn, file_path, file_type):
    table_name = os.path.splitext(os.path.basename(file_path))[0]
    full_table_name = f"{SCHEMA}.{table_name}"

    drop_table_if_exists(conn, table_name)

    # Upload the file based on its type
    if file_type == "shp":
        upload_command = (
            ["ogr2ogr"]
            + OGR2OGR_OPTS
            + ["-nln", full_table_name]
            + DB_OPTS
            + [file_path]
        )
    elif file_type == "geojson":
        upload_command = (
            ["ogr2ogr"]
            + OGR2OGR_OPTS
            + ["-f", "PostgreSQL"]
            + DB_OPTS
            + [file_path, "-nln", full_table_name]
        )
    else:
        logging.error(f"Unsupported file type: {file_type}")
        return False

    for attempt in range(MAX_RETRIES):
        try:
            subprocess.check_call(upload_command)
            logging.info(f"Successfully loaded {file_path} into {full_table_name}")
            return True
        except subprocess.CalledProcessError as e:
            if attempt < MAX_RETRIES - 1:
                logging.warning(
                    f"Attempt {attempt + 1} failed for {file_path}. Retrying in {RETRY_DELAY} seconds..."
                )
                time.sleep(RETRY_DELAY)
            else:
                logging.error(
                    f"Failed to process {file_path} after {MAX_RETRIES} attempts: {e}"
                )
                return False


def group_shapefile_components(blobs):
    """Group Shapefile components together."""
    shapefile_groups = {}
    for blob in blobs:
        name, ext = os.path.splitext(blob.name)
        if ext.lower() in [".shp", ".shx", ".dbf", ".prj"]:
            if name not in shapefile_groups:
                shapefile_groups[name] = []
            shapefile_groups[name].append(blob)
    return shapefile_groups


def process_geojson(conn, blob):
    table_name = generate_table_name(blob.name)
    full_table_name = f"{SCHEMA}.{table_name}"

    file_path = os.path.join("/tmp", os.path.basename(blob.name))
    blob.download_to_filename(file_path)

    upload_command = (
        ["ogr2ogr"]
        + OGR2OGR_OPTS
        + ["-f", "PostgreSQL"]
        + DB_OPTS
        + [file_path, "-nln", full_table_name]
    )

    success = False
    for attempt in range(MAX_RETRIES):
        try:
            subprocess.check_call(
                upload_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            success = True
            break
        except subprocess.CalledProcessError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    os.remove(file_path)
    return success


def process_shapefile(conn, component_blobs):
    shp_blob = next(blob for blob in component_blobs if blob.name.endswith(".shp"))
    table_name = generate_table_name(shp_blob.name)

    temp_dir = os.path.join("/tmp", table_name)
    os.makedirs(temp_dir, exist_ok=True)

    for blob in component_blobs:
        file_ext = os.path.splitext(blob.name)[1]
        file_name = f"{table_name}{file_ext}"
        file_path = os.path.join(temp_dir, file_name)
        blob.download_to_filename(file_path)

    shp_file = f"{table_name}.shp"
    shp_path = os.path.join(temp_dir, shp_file)

    full_table_name = f"{SCHEMA}.{table_name}"

    upload_command = (
        ["ogr2ogr"] + OGR2OGR_OPTS + ["-nln", full_table_name] + DB_OPTS + [shp_path]
    )

    success = False
    for attempt in range(MAX_RETRIES):
        try:
            subprocess.check_call(
                upload_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            success = True
            break
        except subprocess.CalledProcessError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    return success


def load_csv_into_server(conn, file_path, full_table_name):
    """Load a CSV file into the PostgreSQL server."""
    try:
        with open(file_path, "r") as f:
            cursor = conn.cursor()
            # Read and sanitize the header row
            header = f.readline().strip().split(",")
            sanitized_header = [
                re.sub(r"[^a-zA-Z0-9_]", "_", col.strip('"').strip()) for col in header
            ]

            # Ensure column names are unique
            seen = set()
            sanitized_header = [
                col if col not in seen and not seen.add(col) else f"{col}_dup"
                for col in sanitized_header
            ]

            create_table_sql = f"""
            drop table if exists {full_table_name} cascade;
            CREATE TABLE {full_table_name} (
                {','.join([f'"{col}" TEXT' for col in sanitized_header])}
            );
            """
            cursor.execute(create_table_sql)

            # Reset file pointer to beginning
            f.seek(0)

            # Use COPY to load the data into the table
            cursor.copy_expert(f"COPY {full_table_name} FROM STDIN WITH CSV HEADER", f)
            conn.commit()
        return True
    except Exception as e:
        print(f"Error loading CSV into {full_table_name}: {e}")
        conn.rollback()
        return False


def process_csv(conn, blob):
    """Process a CSV file from Google Cloud Storage and load it into the database."""
    # Generate a table name based on the blob name
    table_name = generate_table_name(blob.name)
    full_table_name = f"{SCHEMA}.{table_name}"

    # Download the CSV file to a temporary location
    temp_file_name = f"temp_{table_name}.csv"
    temp_file_path = os.path.join("/tmp", temp_file_name)
    blob.download_to_filename(temp_file_path)

    try:
        # Load the CSV into the database
        success = load_csv_into_server(conn, temp_file_path, full_table_name)
        return success
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def count_processable_files(blobs):
    """Count the number of files that will be processed."""
    count = 0
    shapefile_groups = group_shapefile_components(blobs)
    for blob in blobs:
        if blob.name.endswith(".geojson") or blob.name.endswith(".csv"):
            count += 1
        elif blob.name.endswith(".shp"):
            base_name = os.path.splitext(blob.name)[0]
            if base_name in shapefile_groups:
                count += 1
    return count


def process_file(conn, blob, shapefile_groups, processed_shapefiles):
    """Process a single file and return whether it was processed."""
    if blob.name.endswith(".geojson"):
        return process_geojson(conn, blob)
    elif blob.name.endswith(".shp"):
        base_name = os.path.splitext(blob.name)[0]
        if base_name in shapefile_groups and base_name not in processed_shapefiles:
            success = process_shapefile(conn, shapefile_groups[base_name])
            if success:
                processed_shapefiles.add(base_name)
            return success
    elif blob.name.endswith(".csv"):
        return process_csv(conn, blob)
    return False


def download_and_process_files(bucket, conn, folder_prefix=""):
    """Download and process files from the specified folder and its subfolders in the GCS bucket."""
    blobs = list(bucket.list_blobs(prefix=folder_prefix))
    total_files = count_processable_files(blobs)
    shapefile_groups = group_shapefile_components(blobs)

    processed_shapefiles = set()

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for blob in blobs:
            if blob.name.endswith("/"):  # This is a folder
                continue
            processed = process_file(conn, blob, shapefile_groups, processed_shapefiles)
            if processed:
                pbar.update(1)
            else:
                pbar.total -= 1
                pbar.refresh()


def main(process_entire_bucket=False):
    try:
        # Initialize Google Cloud Storage client
        storage_client = storage.Client(project=PROJECT_NAME)
        bucket = storage_client.bucket(BUCKET_NAME)

        # Connect to the database
        conn = get_db_connection()
        create_schema_if_not_exists(conn)

        if process_entire_bucket:
            print("Processing entire bucket")
            download_and_process_files(bucket, conn)
        else:
            # Process files in the specified folders
            for folder in FOLDERS:
                print(f"Processing folder: {folder}")
                download_and_process_files(bucket, conn, folder)

        print("Processing completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if "conn" in locals() and conn:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process files from Google Cloud Storage bucket"
    )
    parser.add_argument(
        "--full-bucket",
        action="store_true",
        help="Process the entire bucket instead of specific folders",
    )
    args = parser.parse_args()

    main(process_entire_bucket=args.full_bucket)
