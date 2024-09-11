{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['year_']}
    ]
  )
}}

with census_tracts as (
  select
    census_tract as id,
    year_,
    st_transform(geom, 4269) as geom
  from {{ ref('census_tracts_in_city_boundary') }}
)
select
  year_,
  json_build_object('type', 'FeatureCollection', 'features', json_agg(ST_AsGeoJSON(census_tracts.*)::json))
from census_tracts
group by year_
