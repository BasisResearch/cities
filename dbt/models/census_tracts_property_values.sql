{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract_id'], 'unique': true},
    ]
  )
}}

-- Median and total parcel property values aggregated by census tract.

with parcels as (
  select
    parcel_id
    , emv_total
  from {{ ref('parcels_base') }}
)
, census_tracts as (
  select
    census_tract_id
  from {{ ref('census_tracts') }}
)
, parcels_to_census_tracts as (
  select
    parcel_id
    , census_tract_id
  from {{ ref('parcels_to_census_tracts') }}
)
select
  census_tracts.census_tract_id
  , sum(parcels.emv_total) as total_value
  , {{ median('parcels.emv_total') }} as median_value
from
  census_tracts
    left join parcels_to_census_tracts using (census_tract_id)
    left join parcels using (parcel_id)
group by 1
