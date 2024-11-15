{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['zip_code']},
      {'columns': ['zcta']}
    ]
  )
}}

select zip_code, zcta
from {{ source('minneapolis', 'zip_codes_zcta_xref') }}
