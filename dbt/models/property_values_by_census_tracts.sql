-- Median and total parcel property values aggregated by census tract.

with parcels as (
  select
    parcel_id
    , emv_total
  from {{ ref('parcels_base') }}
)
, parcels_to_census_tracts as (
  select
    parcel_id
    , census_tract_id
  from {{ ref('parcels_to_census_tracts') }}
)
select
  parcels_to_census_tracts.census_tract_id
  , sum(parcels.emv_total) as total_value
  , percentile_cont(0.5) within group (order by parcels.emv_total) as median_value
from
  parcels_to_census_tracts using (parcel_id)
    inner join parcels using (parcel_id)
group by census_tracts.census_tract_id
