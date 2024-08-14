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
  , nonres_gro as group_
  , nonres_sub as subgroup
  , nonres_typ as type_category
  , bldg_name as building_name
  , bldg_desc as building_description
  , permit_typ as permit_type
  , permit_val as permit_value
  , sqf as square_feet
  , address
  , geom
  from
    {{ source('minneapolis_old', 'commercial_permits_raw') }}
 where
    co_code = '053'
   and lower(ctu_name) = 'minneapolis'
