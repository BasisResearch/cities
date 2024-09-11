{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['zcta_id', 'year_', 'num_bedrooms']}
    ]
  )
}}

with
fair_market_rents as (select * from {{ ref('stg_fair_market_rents_add_zcta') }})
select
  zcta_id,
  year_::smallint,
  num_bedrooms::smallint,
  avg(rent) as rent
from fair_market_rents
group by 1,2,3
