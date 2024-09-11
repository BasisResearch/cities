{{
  config(
    materialized='table',
  )
}}

with
housing_units as (select * from {{ ref('census_tracts_housing_units') }})
, property_values as (select * from {{ ref('census_tracts_property_values') }})
, distance_to_transit as (select * from {{ ref('census_tracts_distance_to_transit') }})
, parcel_area as (select * from {{ ref('census_tracts_parcel_area') }})
, parking_limits as (select * from {{ ref('census_tracts_parking_limits') }})
, demographics as (select * from {{ ref('demographics') }})
, census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }})

-- Demographic data
, white as (
  select * from demographics
  where name_ = 'B03002_003E' -- white non-hispanic population
)
, population as (
  select * from demographics
  where name_ = 'B01003_001E' -- total population
)
, white_frac as (
  select white.census_tract, white.year_, {{ safe_divide('white.value_', 'population.value_') }} as value_
  from white inner join population using (census_tract, year_)
)
, income as (
  select * from demographics
  where name_ = 'B19013_001E' -- median household income
)
, segregation as (
  select * from demographics
  where description = 'segregation_index_annual_city'
)

, raw_data as (
select
  census_tracts.census_tract::bigint
  , census_tracts.year_::smallint as "year"
  , coalesce(housing_units.num_units, 0) as housing_units
  , property_values.total_value
  , property_values.median_value
  , distance_to_transit.median_distance_to_transit as median_distance
  , distance_to_transit.mean_distance_to_transit as mean_distance
  , parcel_area.parcel_sqm::double precision
  , parcel_area.parcel_mean_sqm::double precision
  , parcel_area.parcel_median_sqm::double precision
  , parking_limits.mean_limit::double precision
  , white_frac.value_ as white
  , income.value_ as income
  , segregation.value_ as segregation
from
  census_tracts
  inner join housing_units using (census_tract_id)
  inner join property_values using (census_tract_id)
  inner join distance_to_transit using (census_tract_id)
  inner join parcel_area using (census_tract_id)
  inner join parking_limits using (census_tract_id)
  left join segregation using (census_tract, year_)
  left join white_frac using (census_tract, year_)
  left join income using (census_tract, year_)
)
, with_std as (
select
  census_tract
  , {{ standardize_cat(['year']) }}
  , {{ standardize_cont(['housing_units', 'total_value', 'median_value',
                         'median_distance', 'mean_distance', 'parcel_sqm',
                         'parcel_mean_sqm', 'parcel_median_sqm', 'white',
                         'income', 'mean_limit', 'segregation' ]) }}
from
  raw_data
)
select * from with_std
