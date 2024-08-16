{% set years = range(2012, 2025) %}

{% for year_ in years %}
select
  zip_code
  , rent_br0
  , rent_br1
  , rent_br2
  , rent_br3
  , rent_br4
  , year as year_
from
  {{ source('minneapolis', 'fair_market_rents_' ~ year_) }}
{% if not loop.last %} union all {% endif %}
{% endfor %}
