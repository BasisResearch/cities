{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['parking_id'], 'unique': true},
      {'columns': ['geom'], 'type': 'gist'}
    ]
  )
}}

with
  stg_parking as (select * from {{ ref('stg_parking') }}),
  stg_parking_to_parcels as (select * from {{ ref('stg_parking_to_parcels') }}),
  parcels as (select * from {{ ref('parcels') }})
select
  stg_parking.*,
  stg_parking_to_parcels.parcel_id,
  parcels.census_block_group_id,
  parcels.census_tract_id,
  parcels.zip_code_id
from
  stg_parking
  left join stg_parking_to_parcels using parking_id
  left join parcels using parcel_id
