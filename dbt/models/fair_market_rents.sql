{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['zip_code_id', 'year_', 'num_bedrooms']}
    ]
  )
}}

with
stg_fair_market_rents_unpivot as (
  select * from {{ ref('stg_fair_market_rents_unpivot') }}
),
zip_codes as (select * from {{ ref('zip_codes') }})
select
  zip_codes.zip_code_id,
  stg_fair_market_rents_unpivot.zip_code,
  stg_fair_market_rents_unpivot.year_::smallint,
  stg_fair_market_rents_unpivot.num_bedrooms::smallint,
  stg_fair_market_rents_unpivot.rent::smallint
from
  stg_fair_market_rents_unpivot
  left join zip_codes
  on stg_fair_market_rents_unpivot.zip_code = zip_codes.zip_code
  and (stg_fair_market_rents_unpivot.year_ || '-01-01')::date <@ zip_codes.valid
