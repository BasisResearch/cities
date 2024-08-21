with
census_tracts_in_city_boundary as (
  select
    census_tract_id
  from {{ ref('census_tracts_in_city_boundary') }}
)
, census_tracts as (
  select
    census_tract_id
    , statefp || countyfp || tractce as census_tract
    , year_
  from {{ ref('census_tracts') }}
  where
    year_ <= 2020
    and census_tract_id in (select * from census_tracts_in_city_boundary)
)
, housing_units as (
  select * from {{ ref('census_tracts_housing_units') }}
)
, property_values as (
  select * from {{ ref('census_tracts_property_values') }}
)
, distance_to_transit as (
  select * from {{ ref('census_tracts_distance_to_transit') }}
)
, parcel_area as (
  select * from {{ ref('census_tracts_parcel_area') }}
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
from
  census_tracts
  inner join housing_units using (census_tract_id)
  inner join property_values using (census_tract_id)
  inner join distance_to_transit using (census_tract_id)
  inner join parcel_area using (census_tract_id)
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
  , {{ standardize(['num_units', 'total_value', 'median_value', 'median_distance_to_transit', 'mean_distance_to_transit', 'parcel_sqm']) }}
from
  raw_data
)
select * from with_std
