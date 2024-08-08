drop table if exists zip_code cascade;

create table zip_code (
    id serial primary key
    , zip_code text not null
    , valid daterange not null
    , geom geometry(MultiPolygon , 4269) not null
);

create index zip_code_geom_idx on zip_code using gist (geom);

insert into zip_code (zip_code , valid , geom)
select
    zcta5ce20
    , '[2020-01-01,)'::daterange
    , geom
from
    zip_raw_2020
union
select
    zcta
    , '[2000-01-01,2020-01-01)'::daterange
    , ST_Transform (geom , 4269)
from
    zip_raw_2000
