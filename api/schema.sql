drop schema if exists api cascade;

create schema api;

create view api.parcels as (
  select * from dbt.parcels
);

create view api.census_tracts as (
  select * from dbt.census_tracts
);

create view api.census_block_groups as (
  select * from dbt.census_block_groups
);

create view api.zip_codes as (
  select * from dbt.zip_codes
);

create view api.emv_in_downtown_west as (
  select dbt.parcels.pin, dbt.parcels.emv_land
    from dbt.parcels
         inner join dbt.neighborhoods
             on st_intersects(st_transform(dbt.parcels.geom, 3857), dbt.neighborhoods.geom)
    where dbt.neighborhoods.name_ = 'Downtown West'
);

drop role if exists web_anon;
create role web_anon nologin;
grant usage on schema api to web_anon;
grant select on all tables in schema api to web_anon;
grant web_anon to postgres;
