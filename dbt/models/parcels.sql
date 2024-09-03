{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['parcel_id'], 'unique': true},
      {'columns': ['valid', 'geom'], 'type': 'gist'}
    ]
  )
}}

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
