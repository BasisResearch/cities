{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['valid']}
    ]
  )
}}

with
lines as (select * from dev.stg_high_frequency_transit_lines_union),
stops as (select * from dev.high_frequency_transit_stops),
select
  st_transform(lines.geom, 4269) as line_geom,
  st_transform(stops.geom, 4269) as stop_geom
from lines inner join stops on lines.valid && stops.valid
where '2020-01-01'::date <@ lines.valid
