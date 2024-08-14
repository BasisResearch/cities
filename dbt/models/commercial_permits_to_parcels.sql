{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['commercial_permit_id']},
      {'columns': ['parcel_id']}
    ]
  )
}}

with
commercial_permits as (
  select
    commercial_permit_id as id
    , daterange(to_date(year_::text, 'YYYY'), to_date(year_::text, 'YYYY'), '[]') as valid
    , geom
  from {{ ref("commercial_permits") }}
)
, parcels as (
  select
    parcel_id as id
    , valid
    , geom
  from {{ ref("parcels_base") }}
)
select
  child_id as commercial_permit_id
  , parent_id as parcel_id
  , valid
  , type_
from {{ tag_regions("commercial_permits", "parcels") }}
