with
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }}),
parcels as (select * from {{ ref('tracts_model_int__parcels_filtered') }})
select
  census_tract_id,
  sum(st_area(parcels.geom)) as parcel_sqm
from
  census_tracts left join parcels using (census_tract_id)
group by 1
