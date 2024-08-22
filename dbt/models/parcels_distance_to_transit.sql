{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['parcel_id'], 'unique': true},
    ]
  )
}}

-- This model calculates the distance from each parcel to the nearest high
-- frequency transit line or stop
with
  parcels as (select * from {{ ref('parcels') }})
 , lines as (select * from {{ ref('high_frequency_transit_lines') }})
 , stops as (select * from {{ ref('high_frequency_transit_stops') }})
 , lines_and_stops as materialized (
  select
    lines.valid * stops.valid as valid
    , st_union(lines.geom, stops.geom) as geom
  from
    lines inner join stops on lines.valid && stops.valid
)
select
  parcels.parcel_id
  , st_distance(parcels.geom, lines_and_stops.geom) as distance
from
  parcels
  inner join lines_and_stops on parcels.valid && lines_and_stops.valid
