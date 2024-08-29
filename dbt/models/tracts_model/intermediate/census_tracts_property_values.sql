{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract_id'], 'unique': true},
    ]
  )
}}

-- Median and total parcel property values aggregated by census tract.

with
parcels as (select * from {{ ref('parcels') }})
, census_tracts as (select * from {{ ref('census_tracts') }})
select
  census_tracts.census_tract_id
  , sum(parcels.emv_total) as total_value
  , {{ median('parcels.emv_total') }} as median_value
from
  census_tracts
    left join parcels using (census_tract_id)
group by 1
