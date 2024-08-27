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
  , "project na"::text as project_name
  , address::text
  , neighborho::text as neighborhood
  , ward::int
  , "downtown y" = 'Y' as is_downtown
  , "housing un"::int as num_housing_units
  , "car parkin"::int as num_car_parking_spaces
  , "bike parki"::int as num_bike_parking_spaces
  , st_transform(geom, {{ var("srid") }}) as geom
from parking_raw
