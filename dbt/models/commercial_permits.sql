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
permits_to_first_parcel as (
  select commercial_permit_id, min(parcel_id) as parcel_id
  from stg_commercial_permits_to_parcels group by 1
),

parcels as (select * from {{ ref('parcels') }})
select
  stg_commercial_permits.*,
  permits_to_first_parcel.parcel_id,
  parcels.census_block_group_id,
  parcels.census_tract_id,
  parcels.zcta_id
from
  stg_commercial_permits
  left join permits_to_first_parcel using (commercial_permit_id)
  left join parcels using (parcel_id)
