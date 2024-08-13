with
census_tracts as (
  select
    census_tract_id
    , statefp
    , countyfp
    , tractce
    , valid

  from {{ ref("census_tracts") }}
)

select
    census_tract_id
    , acs_tract_raw.year_
    , acs_tract_raw.name_
    , acs_tract_raw.value_
from
    acs_tract_raw
    inner join census_tracts
        using (statefp, countyfp, tractce)
 where
   to_date(acs_tract_raw.year_::text , 'YYYY') <@ census_tracts.valid
