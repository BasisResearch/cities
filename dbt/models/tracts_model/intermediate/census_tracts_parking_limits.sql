with
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }}),
parcels_parking_limits as (select * from {{ ref('parcels_parking_limits') }})
select
  census_tract_id,
  avg(limit_numeric) as mean_limit
from census_tracts left join parcels_parking_limits using (census_tract_id)
group by census_tract_id
