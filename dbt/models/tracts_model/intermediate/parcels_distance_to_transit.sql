-- This model calculates the distance from each parcel to the nearest high
-- frequency transit line or stop
with
parcels as (select * from {{ ref('tracts_model_int__parcels_filtered') }}),
lines as (select * from {{ ref('high_frequency_transit_lines') }}),
stops as (select * from {{ ref('high_frequency_transit_stops') }}),
lines_and_stops as materialized (
  select
    lines.valid * stops.valid as valid,
    st_union(lines.geom, stops.geom) as geom,
    lines.geom as line_geom,
    stops.geom as stop_geom
  from
    lines inner join stops on lines.valid && stops.valid
)
select
  parcels.parcel_id,
  parcels.census_tract_id,
  st_distance(parcels.geom, lines_and_stops.geom) as distance,
  st_distance(parcels.geom, lines_and_stops.line_geom) as line_distance,
  st_distance(parcels.geom, lines_and_stops.stop_geom) as stop_distance
from
  parcels
  inner join lines_and_stops on parcels.valid && lines_and_stops.valid
