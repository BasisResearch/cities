{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['parcel_id'], 'unique': true},
    ]
  )
}}

with
  parcels as (
    select
      parcel_id
      , valid
      , geom
    from {{ ref('parcels_base') }}
 )
 , high_frequency_transit_lines as (
    select
      valid
      , geom
    from {{ ref('high_frequency_transit_lines') }}
 )
select
  parcels.parcel_id
  , st_distance(parcels.geom, high_frequency_transit_lines.geom) as distance
from
  parcels
    inner join high_frequency_transit_lines
    on parcels.valid && high_frequency_transit_lines.valid
