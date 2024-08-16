drop schema if exists api cascade;

create schema api;

create view api.acs_tract_wide as (
  select * from dbt.acs_tract_wide
  order by random()
);

drop role if exists web_anon;
create role web_anon nologin;
grant usage on schema api to web_anon;
grant select on all tables in schema api to web_anon;
grant web_anon to postgres;
