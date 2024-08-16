{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract_id', 'year_', 'name_'], 'unique': true},
    ]
  )
}}

with
census_tracts as (
  select
    census_tract_id
    , statefp
    , countyfp
    , tractce
    , valid

  from {{ ref("census_tracts") }}
)
select
    census_tract_id
    , acs_tract_raw.year_
    , acs_tract_raw.name_
    , acs_tract_raw.value_
from
    {{ source('minneapolis_old', 'acs_tract_raw') }}
    inner join census_tracts
        using (statefp, countyfp, tractce)
 where
   to_date(acs_tract_raw.year_::text , 'YYYY') <@ census_tracts.valid
