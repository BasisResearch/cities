{{
  config(
    materialized='table',
  )
}}

with
parcels_parking_limits as (select * from {{ ref('parcels_parking_limits') }}),
parcels_distance_to_transit as (select * from {{ ref('parcels_distance_to_transit') }}),
parcels as (select * from {{ ref('tracts_model_int__parcels_filtered') }}),
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }})
select
  parcels.*,
  census_tracts.census_tract,
  census_tracts.year_,
  parcels_distance_to_transit.distance as distance_to_transit,
  parcels_distance_to_transit.line_distance as distance_to_transit_line,
  parcels_distance_to_transit.stop_distance as distance_to_transit_stop,
  parcels_parking_limits.limit_numeric as limit_con,
  parcels_parking_limits.is_downtown as downtown_yn,
  parcels_parking_limits.is_university as university_yn
from
  parcels
  join census_tracts using (census_tract_id)
  join parcels_parking_limits using (parcel_id)
  join parcels_distance_to_transit using (parcel_id)
