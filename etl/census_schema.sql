drop table if exists census_tract cascade;

create table census_tract (
    id serial primary key
    , statefp text not null
    , countyfp text not null
    , tractce text not null
    , geoidfq text not null
    , valid daterange not null
    , geom geometry(MultiPolygon , 4269) not null
);

create index census_tract_geom_idx on census_tract using gist (geom);

insert into census_tract (statefp , countyfp , tractce , geoidfq , valid , geom)
select
    statefp
    , countyfp
    , tractce
    , affgeoid
    , '[2010-01-01,2020-01-01)'::daterange
    , geom
from
    cb_2018_27_tract_500k
union all
select
    statefp
    , countyfp
    , tractce
    , geoidfq
    , '[2020-01-01,2030-01-01)'::daterange
    , geom
from
    cb_2023_27_tract_500k;

drop table if exists census_block_group cascade;

create table census_block_group (
    id serial primary key
    , statefp text not null
    , countyfp text not null
    , tractce text not null
    , blkgrpce text not null
    , geoidfq text not null
    , tract_id int references census_tract (id)
    , valid daterange not null
    , geom geometry(MultiPolygon , 4269) not null
);

create index census_block_group_geom_idx on census_block_group using gist (geom);

insert into census_block (statefp , countyfp , tractce , blkgrpce , geoidfq , tract_id , valid , geom)
select
    statefp
    , countyfp
    , tractce
    , blkgrpce
    , bg.geoidfq
    , census_tract.id
    , bg.valid
    , bg.geom
from (
    select
        statefp
        , countyfp
        , tractce
        , blkgrpce
        , affgeoid as geoidfq
        , '[2010-01-01,2020-01-01)'::daterange as valid
        , geom
    from
        cb_2018_27_bg_500k
    union all
    select
        statefp
        , countyfp
        , tractce
        , blkgrpce
        , geoidfq
        , '[2020-01-01,2030-01-01)'::daterange as valid
        , geom
    from
        cb_2023_27_bg_500k) as bg
    join census_tract using (statefp , countyfp , tractce)
where
    census_tract.valid && bg.valid;

