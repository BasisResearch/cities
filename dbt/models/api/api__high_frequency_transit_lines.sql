{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['valid']}
    ]
  )
}}

select
  high_frequency_transit_line_id,
  valid,
  st_transform(geom, 4269) as geom,
  st_transform(blue_zone_geom, 4269) as blue_zone_geom,
  st_transform(yellow_zone_geom, 4269) as yellow_zone_geom
from
  {{ ref('high_frequency_transit_lines') }}
