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
  geom,
  blue_zone_geom,
  yellow_zone_geom
from
  {{ ref('high_frequency_transit_lines') }}
