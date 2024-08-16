with census_tracts as (
  select
    census_tract_id
    , geom
  from {{ ref('census_tracts') }}
)
, city_boundary as (
  select
    st_transform(geom, 4269) as geom
  from {{ ref('city_boundary') }}
)
select
  census_tracts.census_tract_id
from
  census_tracts
  , city_boundary
where st_intersects(census_tracts.geom, city_boundary.geom)
      and st_area(st_intersection(census_tracts.geom, city_boundary.geom)) / st_area(census_tracts.geom) > 0.2
