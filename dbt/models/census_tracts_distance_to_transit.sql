{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract_id'], 'unique': true},
    ]
  )
}}

with
  parcels_distance_to_transit as (
    select * from {{ ref('parcels_distance_to_transit') }}
 )
 , census_tracts as (
    select * from {{ ref('census_tracts') }}
 )
 , parcels_to_census_tracts as (
    select * from {{ ref('parcels_to_census_tracts') }}
 )
select
  census_tracts.census_tract_id
  , avg(parcels_distance_to_transit.distance) as mean_distance_to_transit
  , {{ median('parcels_distance_to_transit.distance') }} as median_distance_to_transit
from
  census_tracts
    left join parcels_to_census_tracts using (census_tract_id)
    left join parcels_distance_to_transit using (parcel_id)
group by 1
