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
    st_transform(geom, 4269) as geom
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
  and st_area(st_intersection(all_zip_codes.geom, city_boundary.geom)) / st_area(all_zip_codes.geom) > 0.2
