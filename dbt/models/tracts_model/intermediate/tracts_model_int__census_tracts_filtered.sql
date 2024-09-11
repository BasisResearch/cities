{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['valid', 'geom'], 'type': 'gist'}
    ]
  )
}}

-- Consider only tracts in the city boundary, replace 2020 tracts with 2019
-- tracts, and regenerate the surrogate key.
with census_tracts_in_city_boundary as (
    select *
    from {{ ref('census_tracts_in_city_boundary') }}
    where 2010 < year_ and year_ < 2020
),
census_tracts_union as (
select census_tract, year_, valid, geom from census_tracts_in_city_boundary
union all
select
  census_tract,
  2020 as year_,
  '[2020-01-01,2021-01-01)'::daterange as valid,
  geom
from census_tracts_in_city_boundary where year_ = 2019
)
select
  {{ dbt_utils.generate_surrogate_key(['census_tract', 'year_']) }} as census_tract_id,
  census_tract,
  year_,
  valid,
  geom
from census_tracts_union
