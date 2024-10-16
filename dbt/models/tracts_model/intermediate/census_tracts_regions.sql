with
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }}),
university as (select * from {{ ref('university') }}),
downtown as (select * from {{ ref('downtown') }})
select
  census_tract_id,
  st_area(st_intersection(census_tracts.geom, university.geom)) / st_area(census_tracts.geom) as university_overlap,
  st_area(st_intersection(census_tracts.geom, downtown.geom)) / st_area(census_tracts.geom) as downtown_overlap
from census_tracts, university, downtown
