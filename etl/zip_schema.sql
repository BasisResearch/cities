drop table if exists zip_code;
create table zip_code (
  id serial primary key
  , zip_code text not null
  , year int not null
  , geom geometry(MultiPolygon, 4269) not null
);
create index zip_code_geom_idx on zip_code using gist(geom);
