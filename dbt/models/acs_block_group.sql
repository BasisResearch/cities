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

select
    census_block_group_id
    , year_
    , name_
    , value_
from
    acs_bg_raw
    inner join census_block_groups using (statefp, countyfp, tractce, blkgrpce)
where
    to_date(acs_bg_raw.year_::text , 'YYYY') <@ census_block_groups.valid
