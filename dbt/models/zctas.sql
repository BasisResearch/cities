{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['zcta_id'], 'unique': true},
      {'columns': ['valid', 'geom'], 'type': 'gist'}
    ]
  )
}}

with
zctas as (
select
  zcta,
  '[2020-01-01,)'::daterange as valid,
  geom
from {{ ref('stg_zctas_2020') }}
union all
select
  zcta,
  '[,2020-01-01)'::daterange as valid,
  geom
from {{ ref('stg_zctas_2010') }}
)
select
  {{ dbt_utils.generate_surrogate_key(['zcta', 'valid']) }} as zcta_id,
  zctas.*
from zctas
