{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['residential_permit_id']},
      {'columns': ['census_tract_id']}
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
, census_tracts as (
  select
    census_tract_id as id
    , valid
    , geom
  from {{ ref("census_tracts") }}
)
select
  child_id as residential_permit_id
  , parent_id as census_tract_id
  , valid
  , type_
from {{ tag_regions("residential_permits", "census_tracts") }}
