create extension if not exists postgis;

drop table if exists parcel_geom cascade;

create table parcel_geom (
    id serial primary key
    , geom geometry(MultiPolygon , 26915) not null
);

create index parcel_geom_idx on parcel_geom using gist (geom);

drop table if exists parcel cascade;

create table parcel (
    id serial primary key
    , pid text not null
    , valid daterange not null
    , emv_land numeric
    , emv_building numeric
    , emv_total numeric
    , year_built int
    , sale_date date
    , sale_value numeric
    , geom_id int references parcel_geom (id)
);

comment on column parcel.valid is 'Dates for which this parcel is valid';

comment on column parcel.pid is 'Municipal parcel ID';

comment on column parcel.emv_land is 'Estimated Market Value, land';

comment on column parcel.emv_building is 'Estimated Market Value, buildings';

comment on column parcel.emv_total is 'Estimated Market Value, total (may be more than sum of land and building)';

