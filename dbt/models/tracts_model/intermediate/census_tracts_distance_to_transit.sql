with
parcels_distance_to_transit as (select * from {{ ref('parcels_distance_to_transit') }}),
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }})
select
  census_tracts.census_tract_id,
  avg(parcels_distance_to_transit.distance) as mean_distance_to_transit,
  {{ median('parcels_distance_to_transit.distance') }} as median_distance_to_transit
from
  census_tracts
  left join parcels_distance_to_transit using (census_tract_id)
group by 1
