select
    statefp
    , countyfp
    , tractce
    , blkgrpce
    , year as year_
    , code as name_
    , case when "value" < 0 then null else "value" end as value_
from
    {{ source('minneapolis', 'acs_bg_raw') }}
