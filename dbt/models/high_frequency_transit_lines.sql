with lines as (
  select
    year_
    , geom
  from {{ ref('high_frequency_transit_lines_union') }}
)
, stops as (
  select
    year_
    , geom
  from {{ ref('high_frequency_transit_stops') }}
)
select
  year_ as high_frequency_transit_lines_id
  , year_
  , lines.geom
  -- note units are in meters
  , st_buffer(lines.geom, 106.7) as blue_zone_geom -- 350 feet
  , st_union(st_buffer(lines.geom, 402.3), st_buffer(stops.geom, 804.7)) as yellow_zone_geom -- quarter mile around lines and half mile around stops
from
  lines
    inner join stops using (year_)
