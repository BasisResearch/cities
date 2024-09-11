with census_tracts as (
  select * from {{ ref('census_tracts') }}
)
, city_boundary as (
  select * from {{ ref('city_boundary') }}
)
select
  census_tracts.census_tract_id
  , census_tracts.valid
  , census_tracts.census_tract
  , census_tracts.year_
  , census_tracts.geom
from
  census_tracts
  , city_boundary
where st_intersects(census_tracts.geom, city_boundary.geom)
      and st_area(st_intersection(census_tracts.geom, city_boundary.geom)) / st_area(census_tracts.geom) > 0.9
