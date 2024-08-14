with census_tracts as (
  {% for year_ in var('census_years') %}
select
  {% if year_ == 2010 %}
    state as statefp
    , county as countyfp
    , tract as tractce
    , geo_id as geoidfq
    , '[,2013-01-01)'::daterange as valid
  {% else %}
    statefp
    , countyfp
    , tractce
    , {{ 'geoidfq' if year_ >= 2023 else 'affgeoid' }} as geoidfq
    , '[{{year_}}-01-01,{{ year_ + 1 }}-01-01)'::daterange as valid
{% endif %}
    , geom
from
    {{ source('minneapolis', 'cb_' ~ year_ ~ '_27_tract_500k') }}
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
