{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['parcel_id'], 'unique': true},
      {'columns': ['zip_code_id']}
    ]
  )
}}

with
parcels as (
  select
    parcel_id as id
    , valid
    , ST_Transform(geom, 4269) as geom
  from {{ ref("parcels_base") }}
),
zip_codes as (
  select
    zip_code_id as id
    , valid
    , geom
  from {{ ref("zip_codes") }}
)
select
  child_id as parcel_id
  , parent_id as zip_code_id
  , valid
  , type_
from {{ tag_regions("parcels", "zip_codes") }}
