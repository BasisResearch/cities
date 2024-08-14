{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['zip_code_id'], 'unique': true},
      {'columns': ['geom'], 'type': 'gist'}
    ]
  )
}}

with
zip_codes as (
select
  zip_code,
  '[2020-01-01,)'::daterange as valid,
  geom
from {{ ref('zip_codes_2020') }}
union all
select
  zip_code,
  '[2000-01-01,2020-01-01)'::daterange as valid,
  ST_Transform(geom, 4269) as geom
from {{ ref('zip_codes_2000') }}
)
select
  {{ dbt_utils.generate_surrogate_key(['zip_code', 'valid']) }} as zip_code_id
  , zip_code
  , valid
  , geom
from zip_codes
