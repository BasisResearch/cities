drop schema if exists api cascade;

create schema api;

create view api.demographics as (
  select * from demographics_wide
);

drop role if exists web_anon;
create role web_anon nologin;
grant usage on schema api to web_anon;
grant select on all tables in schema api to web_anon;
grant web_anon to postgres;
