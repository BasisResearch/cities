with
acs_bg_raw as (
  select
    statefp
    , countyfp
    , tractce
    , blkgrpce
    , year
    , code
    , value
  from {{ source('minneapolis', 'acs_bg_raw') }}
)
select
    statefp
    , countyfp
    , tractce
    , blkgrpce
    , year as year_
    , code as name_
    , case when "value" < 0 then null else "value" end as value_
from
    acs_bg_raw
