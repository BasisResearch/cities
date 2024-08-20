{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['residential_permit_id'], 'unique': true},
      {'columns': ['geom'], 'type': 'gist'}
    ]
  )
}}

select
  sde_id::int as residential_permit_id
  , year::int as year_
  , tenure
  , housing_ty as housing_type
  , res_permit as permit_type
  , address
  , name as name_
  , buildings as num_buildings
  , units as num_units
  , age_restri as num_age_restricted_units
  , memory_car as num_memory_care_units
  , assisted as num_assisted_living_units
  , com_off_re = 'Y' as is_commercial_and_residential
  , nullif(sqf, 0) as square_feet
  , public_fun = 'Y' as is_public_funded
  , nullif(permit_val, 0) as permit_value
  , community_ as community_designation
  , notes
  , st_transform(geom, {{ var("srid") }}) as geom
from
    {{ source('minneapolis', 'residential_permits_residentialpermits') }}
where
    co_code = '053'
    and lower(ctu_name) = 'minneapolis'
