{% set num_bedrooms = range(0, 5) %}

with
zip_codes as (select * from {{ ref('zip_codes') }})
, fair_market_rents as (select * from {{ ref('fair_market_rents_union') }})
, fmr_zip as (
  select
    zip_codes.zip_code_id
    {% for bedroom in num_bedrooms %}
    , fair_market_rents.rent_br{{ bedroom }}
    {% endfor %}
    , fair_market_rents.year_
  from
    fair_market_rents
    inner join zip_codes
        on zip_codes.zip_code = fair_market_rents.zip_code
        and zip_codes.valid @> to_date(year_::text , 'YYYY')
)
{% for bedroom in num_bedrooms %}
select
  zip_code_id
  , rent_br{{ bedroom }} as rent
  , {{ bedroom }} as num_bedrooms
  , year_
from fmr_zip
{% if not loop.last %} union all {% endif %}
{% endfor %}
