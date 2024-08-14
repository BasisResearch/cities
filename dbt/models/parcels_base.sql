{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['parcel_id'], 'unique': true},
      {'columns': ['geom'], 'type': 'gist'}
    ]
  )
}}

{% set years = range(2002, 2024) %}
{% set city = 'MINNEAPOLIS' %}
{% set county_id = '053' %}

with parcels as (
  {% for year_ in years %}
  select
  ogc_fid,
  replace(pin, '{{ county_id }}-', '') as pin,
  '[{{ year_ - 1 }}-01-01,{{ year_ }}-01-01)'::daterange as valid,
  nullif(emv_land, 0) as emv_land,
  nullif(emv_bldg, 0) as emv_bldg,
  nullif(emv_total, 0) as emv_total,
  nullif(year_built, 0) as year_built,
  sale_date,
  nullif(sale_value, 0) as sale_value,
  geom
  from minneapolis.parcels{{ year_ }}hennepin
  where upper({{ "city" if year_ < 2018 else "ctu_name" }}) = '{{ city }}'
  {% if not loop.last %}union all{% endif %}
  {% endfor %}
)
select
  {{ dbt_utils.generate_surrogate_key(['ogc_fid', 'valid']) }} as parcel_id
  , pin
  , valid
  , emv_land
  , emv_bldg
  , emv_total
  , year_built
  , sale_date
  , sale_value
  , geom
from parcels
