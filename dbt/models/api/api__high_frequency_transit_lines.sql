{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['valid']}
    ]
  )
}}

with
lines as (select * from {{ ref('high_frequency_transit_lines') }}),
stops as (select * from {{ ref('high_frequency_transit_stops') }})
select
  lines.valid * stops.valid as valid,
  lines.geom as line_geom,
  st_asgeojson(st_transform(lines.geom, 4269))::json as line_geom_json,
  stops.geom as stop_geom,
  st_asgeojson(st_transform(stops.geom, 4269))::json as stop_geom_json
from lines inner join stops on lines.valid && stops.valid
