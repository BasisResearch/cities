{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['description']}
    ]
  )
}}

{% set years = range(2013, 2023) %}

with acs_tract as (
  select
    census_tract_id
    , year_
    , name_
    , value_
  from {{ ref('acs_tract') }}
)

, census_tracts_in_city_boundary as (
  select
    census_tract_id
  from {{ ref("census_tracts_in_city_boundary") }}
)

, census_tracts as (
  select
    census_tract_id
    , substring(geoidfq from 10) as geoidfq
  from {{ ref("census_tracts") }}
  where census_tract_id in (select census_tract_id from census_tracts_in_city_boundary)
)

, acs_variables as (
  select
    "variable"
    , description
  from {{ ref("acs_variables") }}
)

, acs_tract_extended as (
  select
    acs_tract.census_tract_id
    , census_tracts.geoidfq
    , acs_tract.year_
    , acs_tract.name_
    , acs_tract.value_
  from
    acs_tract
    inner join census_tracts using (census_tract_id)
)

, distinct_tracts_and_variables as (
  select distinct
    geoidfq
    , name_
  from acs_tract_extended
)

select
  acs_variables.description
  , distinct_tracts_and_variables.geoidfq as tract_id
{% for year_ in years %}
  , "{{ year_ }}"
{% endfor %}
from
distinct_tracts_and_variables
inner join acs_variables
      on distinct_tracts_and_variables.name_ = acs_variables.variable
{% for year_ in years %}
left join
(select
  geoidfq
  , name_
  , value_ as "{{ year_}}"
from acs_tract_extended
where year_ = {{ year_ }})
using (geoidfq, name_)
{% endfor %}
