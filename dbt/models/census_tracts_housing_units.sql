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
, residential_permits as (
  select
    residential_permit_id
    , year_
    , permit_value
    , num_units
  from {{ ref('residential_permits') }}
)
, residential_permits_to_census_tracts as (
  select
    residential_permit_id
    , census_tract_id
  from {{ ref('residential_permits_to_census_tracts') }}
)
select
  census_tracts.census_tract_id
  , sum(residential_permits.num_units) as num_units
from
  census_tracts
  left join residential_permits_to_census_tracts using (census_tract_id)
  left join residential_permits using (residential_permit_id)
group by 1
