with
  parking_raw as (
    select
      ogc_fid
      , "date"
      , "project na"
      , address
      , neighborho
      , ward
      , "downtown y"
      , "housing un"
      , "car parkin"
      , "bike parki"
      , "year"
      , geom
    from {{ source('minneapolis', 'parking_parcels') }}
  )
select
  ogc_fid as parking_id
  , to_date("year" || '-' || "date", 'YYYY-DD-Mon') as date_
  , "project na" as project_name
  , address
  , neighborho as neighborhood
  , ward
  , "downtown y" = 'Y' as is_downtown
  , "housing un" as num_housing_units
  , "car parkin" as num_car_parking_spaces
  , "bike parki" as num_bike_parking_spaces
  , geom
from parking_raw
