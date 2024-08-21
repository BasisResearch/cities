{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract_id', 'year_', 'name_'], 'unique': true},
    ]
  )
}}

with
census_tracts as (select * from {{ ref("census_tracts") }})
, acs_tract as (select * from {{ ref('acs_tract_clean') }})
select
    census_tract_id
    , acs_tract.year_
    , acs_tract.name_
    , acs_tract.value_
from
    acs_tract
    inner join census_tracts
        using (statefp, countyfp, tractce)
 where
   to_date(acs_tract.year_::text , 'YYYY') <@ census_tracts.valid
