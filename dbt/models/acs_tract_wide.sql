{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['description']}
    ]
  )
}}

{% set years = range(2013, 2023) %}

with
acs_tract as (select * from {{ ref('acs_tract') }})
, acs_variables as (select * from {{ ref('acs_variables') }})
, census_tracts as (select * from {{ ref('census_tracts_in_city_boundary') }})
, acs_tract_filtered as (
  select acs_tract.*, description
  from acs_tract
  inner join census_tracts using (census_tract, year_)
  inner join acs_variables on acs_tract.name_ = acs_variables.variable
)
, distinct_tracts_and_variables as (
  select distinct
    census_tract
    , name_
    , description
  from acs_tract_filtered
)
select
  description
  , census_tract as tract_id
{% for year_ in years %}
  , "{{ year_ }}"
{% endfor %}
from distinct_tracts_and_variables
{% for year_ in years %}
left join
(select
  census_tract
  , name_
  , value_ as "{{ year_}}"
from acs_tract_filtered
where year_ = {{ year_ }})
using (census_tract, name_)
{% endfor %}
