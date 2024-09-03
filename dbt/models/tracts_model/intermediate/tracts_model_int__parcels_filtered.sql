{{
  config(
    materialized='table'
  )
}}

-- Retag parcels with census tracts (because we replaced the 2020 tracts with the 2019 tracts)
with
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }}),
parcels as (select * from {{ ref('parcels') }}),

parcels_tag as (select parcel_id as id, valid, geom from parcels),
census_tracts_tag as (select census_tract_id as id, valid, geom from census_tracts),
parcels_to_census_tracts as (
  select
    child_id as parcel_id,
    parent_id as census_tract_id
  from {{ tag_regions("parcels_tag", "census_tracts_tag") }}
)

select parcels.*, parcels_to_census_tracts.census_tract_id
from parcels join parcels_to_census_tracts using (parcel_id)
