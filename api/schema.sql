drop schema if exists api cascade;

create schema api;

create view api.demographics as (
  select * from api__demographics
);

create view api.census_tracts as (
  select * from api__census_tracts
);

create view api.high_frequency_transit_lines as (
  select * from api__high_frequency_transit_lines
);

do $$
begin
create role web_anon nologin;
exception when duplicate_object then raise notice '%, skipping', sqlerrm using errcode = sqlstate;
end
$$;

grant usage on schema public to web_anon;
grant select on table public.spatial_ref_sys TO web_anon;
grant usage on schema api to web_anon;
grant select on all tables in schema api to web_anon;
grant web_anon to postgres;
