{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract', 'year_', 'name_'], 'unique': true},
    ]
  )
}}

{% docs acs_tract %}

Contains American Community Survey (ACS) demographic data at a census tract
granularity.

The `name_` column contains the name of the demographic variable (e.g.
`B03002_003E`). See `acs_variables` for a mapping of these codes to
human-readable names.

{% enddocs %}

select
  year::smallint as year_,
  code as name_,
  statefp || countyfp || tractce as census_tract,
  case when "value" < 0 then null else "value" end as value_
from {{ source('minneapolis', 'acs_tract_raw') }}
