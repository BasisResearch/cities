{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['parcel_id'], 'unique': true},
      {'columns': ['census_tract_id']}
    ]
  )
}}

with
parcels as (select * from {{ ref("parcels_base") }})
, census_block_groups as (select * from {{ ref("census_block_groups") }})
, parcels_to_census_block_groups as (
  select * from {{ ref("parcels_to_census_block_groups") }}
)
select
  parcels.parcel_id
  , census_block_groups.census_tract_id
from
  parcels
  left join parcels_to_census_block_groups using (parcel_id)
  left join census_block_groups using (census_block_group_id)
