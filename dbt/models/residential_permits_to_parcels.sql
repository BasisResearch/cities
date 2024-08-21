{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['residential_permit_id']},
      {'columns': ['parcel_id']}
    ]
  )
}}

with
residential_permits as (
  select
    residential_permit_id as id
    , daterange(to_date(year_::text, 'YYYY'), to_date(year_::text, 'YYYY'), '[]') as valid
    , geom
  from {{ ref("residential_permits") }}
)
, parcels as (
  select
    parcel_id as id
    , valid
    , geom
  from {{ ref("parcels") }}
)
select
  child_id as residential_permit_id
  , parent_id as parcel_id
  , valid
  , type_
from {{ tag_regions("residential_permits", "parcels") }}
