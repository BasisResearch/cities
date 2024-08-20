with
acs_tract_raw as (
  select
    statefp
    , countyfp
    , tractce
    , year_
    , name_
    , value_
  from {{ source('minneapolis_old', 'acs_tract_raw') }}
)
select
    statefp
    , countyfp
    , tractce
    , year_
    , name_
    , case when value_ < 0 then null else value_ end as value_
from
    acs_tract_raw
