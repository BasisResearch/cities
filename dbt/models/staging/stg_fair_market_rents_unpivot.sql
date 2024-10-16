with
stg_fair_market_rents_dedup as (select * from {{ ref('stg_fair_market_rents_union') }})
select
  stg_fair_market_rents_dedup.zip_code,
  stg_fair_market_rents_dedup.year_,
  x.num_bedrooms,
  x.rent
from
  stg_fair_market_rents_dedup
  cross join lateral (
    values (0, rent_br0),
           (1, rent_br1),
           (2, rent_br2),
           (3, rent_br3),
           (4, rent_br4)
  ) as x(num_bedrooms, rent)
