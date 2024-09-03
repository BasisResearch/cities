{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_block_group', 'year_', 'name_'], 'unique': true},
    ]
  )
}}

select
  year::smallint as year_,
  code as name_,
  statefp || countyfp || tractce || blkgrpce as census_block_group,
  case when "value" < 0 then null else "value" end as value_
from {{ source('minneapolis', 'acs_bg_raw') }}
