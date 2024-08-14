{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['parking_id']},
      {'columns': ['parcel_id']}
    ]
  )
}}

with
  parking as (
    select
      parking_id as id
      , daterange(date_, date_, '[]') as valid
      , ST_Transform(geom, 26915) as geom
    from {{ ref('parking_base') }}
  )
  , parcels as (
    select
      parcel_id as id
      , valid
      , geom
    from {{ ref('parcels_base') }}
  )
select
  child_id as parking_id
  , parent_id as parcel_id
  , valid
  , type_
from {{ tag_regions("parking", "parcels") }}
