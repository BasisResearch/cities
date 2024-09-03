with
  parking as (
    select
      parking_id as id
      , daterange(date_, date_, '[]') as valid
      , geom
    from {{ ref('stg_parking') }}
  )
  , parcels as (
    select
      parcel_id as id
      , valid
      , geom
    from {{ ref('parcels') }}
  )
select
  child_id as parking_id
  , parent_id as parcel_id
  , valid
  , type_
from {{ tag_regions("parking", "parcels") }}
