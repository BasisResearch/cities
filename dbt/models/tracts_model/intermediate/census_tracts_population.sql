-- Population and population density by census tract
with
demographics as (select * from {{ ref('demographics') }}),
population as (
  select * from demographics
  where name_ = 'B01003_001E' -- total population
),
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }})
select
  census_tracts.census_tract_id,
  population.value_ as total_population,
  population.value_ / st_area(census_tracts.geom) as population_density
from
  census_tracts left join population using (census_tract, year_)
