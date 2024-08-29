with
parcels_parking_limits as (select * from {{ ref('parcels_parking_limits') }}),
parcels as (select * from {{ ref('parcels') }})
select
  census_tract_id,
  avg(limit_numeric) as mean_limit
from parcels join parcels_parking_limits using (parcel_id)
group by census_tract_id
