drop schema if exists api cascade;

create schema api;

create view api.demographics as (
  select * from demographics_wide
);

create view api.census_tracts as (
  select
    census_tract,
    year_,
    geom
  from census_tracts_in_city_boundary
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
