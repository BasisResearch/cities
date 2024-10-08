select distinct * from {{ ref('stg_fair_market_rents_unpivot') }}
