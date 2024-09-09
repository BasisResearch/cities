{{
  config(
    materialized='table'
  )
}}

with
usps_migration as (select * from {{ ref('stg_usps_migration_unpivot') }}),
zctas as (select * from {{ ref('zctas') }}),
zip_codes_to_zctas as (select * from {{ ref('zip_codes_to_zctas') }})
select
  usps_migration.*,
  zctas.zcta_id
from
  usps_migration
  left join zip_codes_to_zctas using (zip_code)
  left join zctas
  on zip_codes_to_zctas.zcta = zctas.zcta
  and usps_migration.date_ <@ zctas.valid
