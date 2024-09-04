with
stg_fair_market_rents_unpivot as (
  select * from {{ ref('stg_fair_market_rents_unpivot') }}
),
zip_codes_to_zctas as (select * from {{ ref('zip_codes_to_zctas') }}),
zctas as (select * from {{ ref('zctas') }})
select
  stg_fair_market_rents_unpivot.zip_code,
  stg_fair_market_rents_unpivot.year_::smallint,
  stg_fair_market_rents_unpivot.num_bedrooms::smallint,
  stg_fair_market_rents_unpivot.rent::smallint,
  zctas.zcta_id
from
  stg_fair_market_rents_unpivot
  left join zip_codes_to_zctas using (zip_code)
  left join zctas
  on zip_codes_to_zctas.zcta = zctas.zcta
  and (stg_fair_market_rents_unpivot.year_ || '-01-01')::date <@ zctas.valid
