with
census_tracts as (
  select
    census_tract_id
    , statefp || countyfp || tractce as census_tract
    , year_
  from {{ ref('census_tracts') }}
)
, census_tracts_housing_units as (
  select
    census_tract_id
    , num_units
  from {{ ref('census_tracts_housing_units') }}
)
, census_tracts_property_values as (
  select
    census_tract_id
    , median_value
    , total_value
  from {{ ref('census_tracts_property_values') }}
)
, census_tracts_distance_to_transit as (
  select
    census_tract_id
    , median_distance_to_transit
    , mean_distance_to_transit
  from {{ ref('census_tracts_distance_to_transit') }}
)
, census_tracts_parcel_area as (
  select
    census_tract_id
    , parcel_sqm
  from {{ ref('census_tracts_parcel_area') }}
)
, raw_data as (
select
  census_tracts.census_tract
  , census_tracts.year_
  , coalesce(census_tracts_housing_units.num_units, 0) as num_units
  , census_tracts_property_values.total_value
  , census_tracts_property_values.median_value
  , census_tracts_distance_to_transit.median_distance_to_transit
  , census_tracts_distance_to_transit.mean_distance_to_transit
  , census_tracts_parcel_area.parcel_sqm
from
  census_tracts_housing_units
  inner join census_tracts_property_values using(census_tract_id)
  inner join census_tracts_distance_to_transit using (census_tract_id)
  inner join census_tracts_parcel_area using (census_tract_id)
  inner join census_tracts using (census_tract_id)
where
  census_tracts.year_ <= 2020
  and census_tracts.census_tract_id in (select census_tract_id from {{ ref('census_tracts_in_city_boundary') }})
)
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
