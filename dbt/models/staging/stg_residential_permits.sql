select
  sde_id::int as residential_permit_id
  , year::smallint as year_
  , tenure::text
  , housing_ty::text as housing_type
  , res_permit::text as permit_type
  , address::text
  , name::text as name_
  , buildings::smallint as num_buildings
  , units::smallint as num_units
  , age_restri::smallint as num_age_restricted_units
  , memory_car::smallint as num_memory_care_units
  , assisted::smallint as num_assisted_living_units
  , com_off_re = 'Y' as is_commercial_and_residential
  , nullif(sqf, 0)::int as square_feet
  , public_fun = 'Y' as is_public_funded
  , nullif(permit_val, 0)::int as permit_value
  , community_::text as community_designation
  , notes::text
  , st_transform(geom, {{ var("srid") }}) as geom
from
    {{ source('minneapolis', 'residential_permits_residentialpermits') }}
where
    co_code = '053'
    and lower(ctu_name) = 'minneapolis'
