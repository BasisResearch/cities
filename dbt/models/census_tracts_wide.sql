{{
  config(
    materialized='table',
  )
}}

with
in_city_boundary as (select * from {{ ref('census_tracts_in_city_boundary') }})
, housing_units as (select * from {{ ref('census_tracts_housing_units') }})
, property_values as (select * from {{ ref('census_tracts_property_values') }})
, distance_to_transit as (select * from {{ ref('census_tracts_distance_to_transit') }})
, parcel_area as (select * from {{ ref('census_tracts_parcel_area') }})
, parking_limits as (select * from {{ ref('census_tracts_parking_limits') }})
, acs_tract as (select * from {{ ref('acs_tract') }})
, census_tracts as (
  select *
  from {{ ref('census_tracts') }}
  where
    year_ <= 2020
    and census_tract_id in (select census_tract_id from in_city_boundary)
)

-- Fill in data for 2011, 2012 using closest available year. Replace 2020 data
-- with 2019 data to avoid pandemic effects.
, acs_replace_years as (
  select * from acs_tract where year_ != 2020
  union all
  select census_tract, 2020 as year_, name_, value_
  from acs_tract where year_ = 2019
  union all
  -- select * from acs_tract
  -- union all
  select census_tract, 2011 as year_, name_, value_
  from acs_tract where year_ = 2013
  union all
  select census_tract, 2012 as year_, name_, value_
  from acs_tract where year_ = 2013
)
, white as (
  select * from acs_replace_years
  where name_ = 'B03002_003E' -- white non-hispanic population
)
, population as (
  select * from acs_replace_years
  where name_ = 'B01003_001E' -- total population
)
, white_frac as (
  select white.census_tract, white.year_, {{ safe_divide('white.value_', 'population.value_') }} as value_
  from white inner join population using (census_tract, year_)
)
, income as (
  select * from acs_replace_years
  where name_ = 'B19013_001E' -- median household income
)
, raw_data as (
select
  census_tracts.census_tract
  , census_tracts.year_
  , coalesce(housing_units.num_units, 0) as num_units
  , property_values.total_value
  , property_values.median_value
  , distance_to_transit.median_distance_to_transit
  , distance_to_transit.mean_distance_to_transit
  , parcel_area.parcel_sqm
  , parking_limits.mean_limit
  , white_frac.value_ as white
  , income.value_ as income
from
  census_tracts
  inner join housing_units using (census_tract_id)
  inner join property_values using (census_tract_id)
  inner join distance_to_transit using (census_tract_id)
  inner join parcel_area using (census_tract_id)
  inner join parking_limits using (census_tract_id)
  left join white_frac using (census_tract, year_)
  left join income using (census_tract, year_)
)
, with_std as (
select
  census_tract
  , year_
  , num_units
  , total_value
  , median_value
  , median_distance_to_transit
  , mean_distance_to_transit
  , parcel_sqm
  , white
  , income
  , {{ standardize(['num_units', 'total_value', 'median_value',
                    'median_distance_to_transit', 'mean_distance_to_transit',
                    'parcel_sqm', 'white', 'income' ]) }}
from
  raw_data
)
select * from with_std
