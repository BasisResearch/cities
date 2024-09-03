{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['parcel_id'], 'unique': true},
      {'columns': ['valid', 'geom'], 'type': 'gist'}
    ]
  )
}}

{% doc parcels %}

Contains the geometry and metadata for all parcels in the city of Minneapolis.

Notes:
- Parcels data is released yearly. Parcels are considered valid for the year they were released.
- Parcels are filtered to only include those in Minneapolis.
- `emv_total`, `emv_bldg`, `emv_land`, `year_built`, and `sale_value` are treated as missing if they are 0.
- `sale_date` is treated as missing if it is equal to `1899-12-30`.
- `pin` is the county-assigned parcel identification number. The county prefix '053-' is removed.
- Duplicate rows are removed. Note that this is based on the entire row, not just the `pin`. There may still be duplicate `pin, year_` pairs.

{% enddoc %}

with
parcels as (select * from {{ ref('stg_parcels') }}),
to_zip_codes as (select * from {{ref('stg_parcels_to_zip_codes')}}),
to_census_bgs as (select * from {{ref('stg_parcels_to_census_block_groups')}}),
census_bgs as (select * from {{ref('census_block_groups')}})
select
  parcels.*
  , to_zip_codes.zip_code_id
  , to_census_bgs.census_block_group_id
  , census_bgs.census_tract_id
from
  parcels
  left join to_zip_codes using (parcel_id)
  left join to_census_bgs using (parcel_id)
  left join census_bgs using (census_block_group_id)
