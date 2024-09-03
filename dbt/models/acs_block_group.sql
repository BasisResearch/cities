{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_block_group', 'year_', 'name_'], 'unique': true},
    ]
  )
}}

{% docs acs_block_group %}

Contains American Community Survey (ACS) demographic data at a census block
group granularity.

The `name_` column contains the name of the demographic variable (e.g.
`B03002_003E`). See `acs_variables` for a mapping of these codes to
human-readable names.

{% enddocs %}

select
  year::smallint as year_,
  code as name_,
  statefp || countyfp || tractce || blkgrpce as census_block_group,
  case when "value" < 0 then null else "value" end as value_
from {{ source('minneapolis', 'acs_bg_raw') }}
