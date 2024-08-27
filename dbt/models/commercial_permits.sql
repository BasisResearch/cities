{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['commercial_permit_id'], 'unique': true},
      {'columns': ['geom'], 'type': 'gist'}
    ]
  )
}}

select
  sde_id as commercial_permit_id
  , year::int as year_
  , nonres_gro::text as group_
  , nonres_sub::text as subgroup
  , nonres_typ::text as type_category
  , bldg_name::text as building_name
  , bldg_desc::text as building_description
  , permit_typ::text as permit_type
  , permit_val::int as permit_value
  , nullif(sqf, 0)::int as square_feet
  , address::text
  , st_transform(geom, {{ var("srid") }}) as geom
from
    {{ source('minneapolis', 'commercial_permits_nonresidentialconstruction') }}
 where
    co_code = '053'
   and lower(ctu_name) = 'minneapolis'
