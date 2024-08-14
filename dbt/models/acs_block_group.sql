with
census_block_groups as (
    select
        census_block_group_id
        , statefp
        , countyfp
        , tractce
        , blkgrpce
        , valid
    from
        {{ ref('census_block_groups') }}
)
, acs_bg as (
  select
    statefp
    , countyfp
    , tractce
    , blkgrpce
    , year_
    , name_
    , value_
    from
        {{ source('minneapolis_old', 'acs_bg_raw') }}
)
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
