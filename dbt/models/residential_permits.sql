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
  sde_id as residential_permit_id
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
  , sqf as square_feet
  , public_fun = 'Y' as is_public_funded
  , permit_val as permit_value
  , community_ as community_designation
  , notes
  , geom
from
    {{ source('minneapolis_old', 'residential_permits_raw') }}
where
    co_code = '053'
    and lower(ctu_name) = 'minneapolis'
