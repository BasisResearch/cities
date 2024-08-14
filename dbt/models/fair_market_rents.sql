{% set num_bedrooms = range(0, 5) %}

with
zip_codes as (
  select
    zip_code_id
    , zip_code
    , valid
  from {{ ref('zip_codes') }}
)
, fmr_zip as (
  select
    zip_codes.zip_code_id
    {% for bedroom in num_bedrooms %}
    , fair_market_rents_raw.rent_br{{ bedroom }}
    {% endfor %}
    , fair_market_rents_raw.year_
  from
    {{ source('minneapolis_old', 'fair_market_rents_raw') }}
    inner join zip_codes
        on zip_codes.zip_code = fair_market_rents_raw.zip
        and zip_codes.valid @> to_date(year_::text , 'YYYY')
)
{% for bedroom in num_bedrooms %}
select
  zip_code_id
  , rent_br{{ bedroom }} as rent
  , 0 as num_bedrooms
  , year_
from fmr_zip
{% if not loop.last %} union all {% endif %}
{% endfor %}
