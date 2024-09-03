{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['commercial_permit_id'], 'unique': true},
      {'columns': ['geom'], 'type': 'gist'}
    ]
  )
}}

with
stg_commercial_permits as (select * from {{ ref('stg_commercial_permits') }}),
stg_commercial_permits_to_parcels as (select * from {{ ref('stg_commercial_permits_to_parcels') }}),
parcels as (select * from {{ ref('parcels') }})
select
  stg_commercial_permits.*,
  stg_commercial_permits_to_parcels.parcel_id,
  parcels.census_block_group_id,
  parcels.census_tract_id,
  parcels.zip_code_id
from
  stg_commercial_permits
  left join stg_commercial_permits_to_parcels using commercial_permit_id
  left join parcels using parcel_id
