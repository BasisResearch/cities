with
parcels as (
  select
    parcel_id as id
    , valid
    , geom
  from {{ ref('stg_parcels_base') }}
),
census_block_groups as (
  select
    census_block_group_id as id
    , valid
    , geom
  from {{ ref('census_block_groups') }}
)
select
  child_id as parcel_id
  , parent_id as census_block_group_id
  , valid
  , type_
from {{ tag_regions("parcels", "census_block_groups") }}
