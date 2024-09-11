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
  stg_parking_to_first_parcel as (
    select parking_id, min(parcel_id) as parcel_id
    from stg_parking_to_parcels group by 1
  ),
  parcels as (select * from {{ ref('parcels') }})
select
  stg_parking.*,
  stg_parking_to_first_parcel.parcel_id,
  parcels.census_block_group_id,
  parcels.census_tract_id,
  parcels.zcta_id
from
  stg_parking
  left join stg_parking_to_first_parcel using (parking_id)
  left join parcels using (parcel_id)
