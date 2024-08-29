with
census_tracts as (select * from {{ ref('census_tracts') }}),
parcels as (select * from {{ ref('parcels') }})
select
  census_tract_id,
  sum(st_area(parcels.geom)) as parcel_sqm
from
  census_tracts left join parcels using (census_tract_id)
group by 1
