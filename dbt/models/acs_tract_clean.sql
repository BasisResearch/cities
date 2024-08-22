with
acs_tract_raw as (
  select
    statefp
    , countyfp
    , tractce
    , year
    , code
    , value
  from {{ source('minneapolis', 'acs_tract_raw') }}
)
select
    statefp
    , countyfp
    , tractce
    , year as year_
    , code as name_
    , case when "value" < 0 then null else "value" end as value_
from
    acs_tract_raw