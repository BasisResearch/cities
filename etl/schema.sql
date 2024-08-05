create extension if not exists postgis;

drop table if exists parcel_geom cascade;
create table parcel_geom (
  parcel_geom_id serial primary key
  , parcel_geom_data geometry(MultiPolygon, 26915) not null
);
create index parcel_geom_data_idx on parcel_geom using gist(parcel_geom_data);

drop table if exists parcel;
create table parcel (
  parcel_pk serial primary key
  , parcel_id text
  , parcel_year int not null

  , parcel_emv_land numeric     -- Estimated Market Value, land
  , parcel_emv_building numeric -- Estimated Market Value, building
  , parcel_emv_total numeric    -- Estimated Market Value, total (may be more than sum of land and building)
  , parcel_year_built int
  , parcel_sale_date date
  , parcel_sale_value numeric

  , parcel_geom_id int references parcel_geom(parcel_geom_id)
);
