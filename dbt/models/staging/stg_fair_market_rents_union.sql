{% set years = range(2012, 2025) %}

{% for year_ in years %}
select
  zip_code
  , replace(rent_br0, '.00', '') as rent_br0
  , replace(rent_br1, '.00', '') as rent_br1
  , replace(rent_br2, '.00', '') as rent_br2
  , replace(rent_br3, '.00', '') as rent_br3
  , replace(rent_br4, '.00', '') as rent_br4
  , year as year_
from
  {{ source('minneapolis', 'fair_market_rents_' ~ year_) }}
{% if not loop.last %} union all {% endif %}
{% endfor %}
