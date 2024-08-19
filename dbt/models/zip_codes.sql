{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['zip_code_id'], 'unique': true},
      {'columns': ['valid', 'geom'], 'type': 'gist'}
    ]
  )
}}

with city_boundary as (
  select
    geom
  from
    {{ ref('city_boundary') }}
)
select
    all_zip_codes.zip_code_id
    , all_zip_codes.zip_code
    , all_zip_codes.valid
    , all_zip_codes.geom
from
  {{ ref('all_zip_codes') }} as all_zip_codes,
  city_boundary
where
  st_intersects(all_zip_codes.geom, city_boundary.geom)
