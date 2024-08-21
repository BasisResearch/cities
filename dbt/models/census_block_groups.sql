{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_block_group_id'], 'unique': true},
      {'columns': ['geom'], 'type': 'gist'},
      {'columns': ['valid', 'geom'], 'type': 'gist'}
    ]
  )
}}

with
census_tracts as (select * from {{ ref("census_tracts") }}),
census_block_groups as (
  {% for year_ in var('census_years') %}
  select
  {% if year_ == 2010 %}
    state as statefp
    , county countyfp
    , tract as tractce
    , blkgrp as blkgrpce
    , geo_id as geoidfq
    , '[,2013-01-01)'::daterange as valid -- use 2010 data for all years before 2013
  {% else %}
    statefp
    , countyfp
    , tractce
    , blkgrpce
    , {{ 'geoidfq' if year_ >= 2023 else 'affgeoid' }} as geoidfq
    , '[{{ year_ }}-01-01,{{ year_ + 1 }}-01-01)'::daterange as valid
  {% endif %}
  , {{ year_ }} as year_
  , st_transform(geom, {{ var("srid") }}) as geom
  from
  {{ source('minneapolis', 'census_cb_' ~ year_ ~ '_27_bg_500k') }}
  {% if not loop.last %}union all{% endif %}
  {% endfor %}
),
census_block_groups_with_tracts as (
  select
    census_block_groups.statefp
    , census_block_groups.countyfp
    , census_block_groups.tractce
    , census_block_groups.blkgrpce
    , census_block_groups.geoidfq
    , census_tracts.census_tract_id
    , (census_block_groups.valid * census_tracts.valid) as valid
    , census_block_groups.geom
  from census_block_groups
       inner join census_tracts using (statefp, countyfp, tractce)
  where
    census_tracts.valid && census_block_groups.valid
)
select
  {{ dbt_utils.generate_surrogate_key(['geoidfq', 'valid']) }} as census_block_group_id,
  *
from census_block_groups_with_tracts
