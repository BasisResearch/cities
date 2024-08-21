{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_block_group_id', 'year_', 'name_'], 'unique': true},
    ]
  )
}}

with
census_block_groups as (select * from {{ ref('census_block_groups') }})
, acs_bg as (select * from {{ ref('acs_block_group_clean') }})
select
    census_block_groups.census_block_group_id
    , acs_bg.year_
    , acs_bg.name_
    , acs_bg.value_
from
    acs_bg
    inner join census_block_groups using (statefp, countyfp, tractce, blkgrpce)
where
    to_date(acs_bg.year_::text , 'YYYY') <@ census_block_groups.valid
