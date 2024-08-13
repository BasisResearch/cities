with census_tracts as (
{% for year_ in var('census_years') %}
select
    {{ 'statefp' if year_ >= 2013 else 'state' }} as statefp
    , {{ 'countyfp' if year_ >= 2013 else 'county' }} as countyfp
    , {{ 'tractce' if year_ >= 2013 else 'tract' }} as tractce
    , {{ 'geoidfq' if year_ >= 2023 else
         'affgeoid' if year_ >= 2013 else
         'geo_id' }} as geoidfq
    , '[{{year_}}-01-01,{{ year_ + 1 }}-01-01)'::daterange as valid
    , geom
from
    minneapolis.cb_{{ year_ }}_27_tract_500k
{% if not loop.last %}union all{% endif %}
{% endfor %}
)
select
    {{ dbt_utils.generate_surrogate_key(['geoidfq', 'valid']) }} as census_tract_id
    , statefp
    , countyfp
    , tractce
    , geoidfq
    , valid
    , geom
from
     census_tracts
