insert into acs_tract
select
    id
    , year_
    , name_
    , value_
from
    acs_tract_raw as t1
    join census_tract as t2 on t1.statefp = t2.statefp
        and t1.countyfp = t2.countyfp
        and t1.tractce = t2.tractce
        and to_date(t1.year_::text , 'YYYY') <@ t2.valid;

insert into acs_bg
select
    id
    , year_
    , name_
    , value_
from
    acs_bg_raw as t1
    join census_bg as t2 on t1.statefp = t2.statefp
        and t1.countyfp = t2.countyfp
        and t1.tractce = t2.tractce
        and t1.blkgrpce = t2.blkgrpce
        and to_date(t1.year_::text , 'YYYY') <@ t2.valid;

