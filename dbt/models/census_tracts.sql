{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract_id'], 'unique': true},
      {'columns': ['valid', 'geom'], 'type': 'gist'}
      {'columns': ['year']}
    ]
  )
}}

with census_tracts_union as (
  {% for year_ in var('census_years') %}
select
  {% if year_ == 2010 %}
    state as statefp
    , county as countyfp
    , tract as tractce
    , geo_id as geoidfq
  {% else %}
    statefp
    , countyfp
    , tractce
    , {{ 'geoidfq' if year_ >= 2023 else 'affgeoid' }} as geoidfq
  {% endif %}
    , '[{{year_}}-01-01,{{ year_ + 1 }}-01-01)'::daterange as valid
    , {{ year_ }} as year_
    , st_transform(geom, {{ var("srid") }}) as geom
from
    {{ source('minneapolis', 'census_cb_' ~ year_ ~ '_27_tract_500k') }}
{% if not loop.last %}union all{% endif %}
{% endfor %}
),
years_2011_2012 as (
  select
    statefp
    , countyfp
    , tractce
    , geoidfq
    , '[2011-01-01,2012-01-01)'::daterange as valid
    , 2011 as year_
    , geom
  from census_tracts_union
  where year_ = 2010
  union all
  select
    statefp
    , countyfp
    , tractce
    , geoidfq
    , '[2012-01-01,2013-01-01)'::daterange as valid
    , 2012 as year_
    , geom
  from census_tracts_union
  where year_ = 2010
),
add_2011_2012 as (
  select *
  from census_tracts_union
  union all
  select *
  from years_2011_2012
),
with_census_tract as (
  select *, statefp || countyfp || tractce as census_tract
  from add_2011_2012
)
select
    {{ dbt_utils.generate_surrogate_key(['geoidfq', 'year_']) }} as census_tract_id, *
from
     with_census_tract
