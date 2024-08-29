-- Median and total parcel property values aggregated by census tract.
with
parcels as (select * from {{ ref('tracts_model_int__parcels_filtered') }}),
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }})
select
  census_tracts.census_tract_id,
  sum(parcels.emv_total) as total_value,
  {{ median('parcels.emv_total') }} as median_value
from
  census_tracts left join parcels using (census_tract_id)
group by 1
