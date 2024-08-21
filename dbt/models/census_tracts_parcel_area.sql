{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract_id'], 'unique': true},
    ]
  )
}}

with census_tracts as (
  select
    census_tract_id
  from {{ ref('census_tracts') }}
)
, parcels as (
  select
    parcel_id
    , geom
  from {{ ref('parcels_base') }}
)
, parcels_to_census_tracts as (
  select
    parcel_id
    , census_tract_id
  from {{ ref('parcels_to_census_tracts') }}
)
select
  census_tract_id
  , sum(st_area(parcels.geom)) as parcel_sqm
from
  census_tracts
    left join parcels_to_census_tracts using (census_tract_id)
    left join parcels using (parcel_id)
group by 1
