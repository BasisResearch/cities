{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['high_frequency_transit_line_id'], 'unique': true},
      {'columns': ['valid', 'geom'], 'type': 'gist'},
    ]
  )
}}

with lines as (select * from {{ ref('high_frequency_transit_lines_union') }})
, stops as (select * from {{ ref('high_frequency_transit_stops') }})
, lines_and_stops as (
  select
    lines.valid * stops.valid as valid
    , lines.geom as line_geom
    , stops.geom as stop_geom
  from
    lines
      inner join stops on lines.valid && stops.valid
)
select
  {{ dbt_utils.generate_surrogate_key(['valid']) }} as high_frequency_transit_line_id
  , valid
  , line_geom as geom
  -- note units are in meters
  , st_buffer(line_geom, 106.7) as blue_zone_geom -- 350 feet
  , st_union(st_buffer(line_geom, 402.3), st_buffer(stop_geom, 804.7)) as yellow_zone_geom -- quarter mile around lines and half mile around stops
from
  lines_and_stops
