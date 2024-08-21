{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract_id'], 'unique': true},
    ]
  )
}}

with census_tracts as (
  select * from {{ ref('census_tracts') }}
)
, residential_permits as (
  select * from {{ ref('residential_permits') }}
)
, residential_permits_to_census_tracts as (
  select * from {{ ref('residential_permits_to_census_tracts') }}
)
select
  census_tracts.census_tract_id
  , sum(residential_permits.num_units) as num_units
from
  census_tracts
  left join residential_permits_to_census_tracts using (census_tract_id)
  left join residential_permits using (residential_permit_id)
group by 1
