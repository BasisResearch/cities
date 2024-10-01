begin;
drop schema if exists api cascade;

create schema api;

create view api.demographics as (
  select * from api__demographics
);

create view api.census_tracts as (
  select * from api__census_tracts
);

create function api.high_frequency_transit_lines() returns setof dev.api__high_frequency_transit_lines as $$
  select * from dev.api__high_frequency_transit_lines
$$ language sql;

create function api.high_frequency_transit_lines(
       blue_zone_radius double precision,
       yellow_zone_line_radius double precision,
       yellow_zone_stop_radius double precision
) returns table (
    valid daterange,
    geom geometry(LineString, 4269),
    blue_zone_geom geometry(LineString, 4269),
    yellow_zone_geom geometry(Geometry, 4269)
) as $$
  with
  lines as (select * from dev.stg_high_frequency_transit_lines_union),
  stops as (select * from dev.high_frequency_transit_stops),
  lines_and_stops as (
    select
      lines.valid * stops.valid as valid,
      lines.geom as line_geom,
      stops.geom as stop_geom
    from lines inner join stops on lines.valid && stops.valid
  )
  select
    valid,
    st_transform(line_geom, 4269) as geom,
    st_transform(st_buffer(line_geom, blue_zone_radius), 4269) as blue_zone_geom,
    st_transform(st_union(st_buffer(line_geom, yellow_zone_line_radius), st_buffer(stop_geom, yellow_zone_stop_radius)), 4269) as yellow_zone_geom
  from lines_and_stops
$$ language sql;

do $$
begin
create role web_anon nologin;
exception when duplicate_object then raise notice '%, skipping', sqlerrm using errcode = sqlstate;
end
$$;

grant all on schema public to web_anon;
grant all on schema dev to web_anon;
grant select on table public.spatial_ref_sys TO web_anon;
grant usage on schema api to web_anon;
grant all on all tables in schema api to web_anon;
grant all on all functions in schema api to web_anon;
grant all on schema api to web_anon;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dev TO web_anon;
GRANT ALL PRIVILEGES ON ALL functions IN SCHEMA dev TO web_anon;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA api TO web_anon;
GRANT ALL PRIVILEGES ON ALL functions IN SCHEMA api TO web_anon;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO web_anon;
GRANT ALL PRIVILEGES ON ALL functions IN SCHEMA public TO web_anon;
grant web_anon to postgres;
commit;
