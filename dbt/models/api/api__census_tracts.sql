{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['year_']}
    ]
  )
}}

with census_tracts as (select * from {{ ref('census_tracts_in_city_boundary') }})
select
  census_tract
  , year_
  , st_transform(geom, 4269) as geom
from
  census_tracts
