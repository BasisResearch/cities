with
parking_raw as (select * from {{ source('minneapolis', 'parking_parcels') }})
select
  ogc_fid as parking_id
  , to_date("year" || '-' || "date", 'YYYY-DD-Mon') as date_
  , "project na"::text as project_name
  , address::text
  , neighborho::text as neighborhood
  , ward::smallint
  , "downtown y" = 'Y' as is_downtown
  , "housing un"::smallint as num_housing_units
  , "car parkin"::smallint as num_car_parking_spaces
  , replace("bike parki", ',', '')::smallint as num_bike_parking_spaces
  , st_transform(geom, {{ var("srid") }}) as geom
from parking_raw
