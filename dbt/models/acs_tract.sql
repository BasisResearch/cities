{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract', 'year_', 'name_'], 'unique': true},
    ]
  )
}}

select
  year::smallint as year_,
  code as name_,
  statefp || countyfp || tractce as census_tract,
  case when "value" < 0 then null else "value" end as value_
from {{ source('minneapolis', 'acs_tract_raw') }}
