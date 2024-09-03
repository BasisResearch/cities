{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['zip_code_id'], 'unique': true},
      {'columns': ['valid', 'geom'], 'type': 'gist'}
    ]
  )
}}

{% docs zip_codes %}

Contains the geometry and metadata for all zip code tabulation areas (ZCTAs) in
the United States.

{% enddocs %}

with
zip_codes as (
select
  zip_code,
  '[2020-01-01,)'::daterange as valid,
  geom
from {{ ref('stg_zip_codes_2020') }}
union all
select
  zip_code,
  '[,2020-01-01)'::daterange as valid,
  geom
from {{ ref('stg_zip_codes_2010') }}
)
select
  {{ dbt_utils.generate_surrogate_key(['zip_code', 'valid']) }} as zip_code_id,
  zip_codes.*
from zip_codes
