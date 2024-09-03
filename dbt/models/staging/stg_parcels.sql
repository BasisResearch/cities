{% set years = range(2002, 2024) %}
{% set city = 'MINNEAPOLIS' %}
{% set county_id = '053' %}

with
-- This is a union of all the parcels from the years 2002 to 2023
parcels_union as (
  {% for year_ in years %}
  select
  ogc_fid,
  replace(pin, '{{ county_id }}-', '') as pin,

  -- parcels are a year-end snapshot, named after the year they cover
  '[{{ year_ }}-01-01,{{ year_ + 1 }}-01-01)'::daterange as valid,
  nullif(emv_land, 0)::int as emv_land,
  nullif(emv_bldg, 0)::int as emv_bldg,
  nullif(emv_total, 0)::int as emv_total,
  nullif(year_built, 0)::int as year_built,
  nullif(sale_date, '1899-12-30'::date) as sale_date,
  nullif(sale_value, 0)::int as sale_value,
  st_transform(geom, {{ var("srid") }}) as geom
  from {{ source('minneapolis', 'parcels_shp_plan_regonal_' ~ year_ ~ '_parcels' ~ year_ ~ 'hennepin') }}
  where upper({{ "city" if year_ < 2018 else "ctu_name" }}) = '{{ city }}'
  {% if not loop.last %}union all{% endif %}
  {% endfor %}
),

-- Some of the parcel datasets contain exact duplicates that we remove. Note
-- that duplicate pin/year pairs may remain.
parcels_distinct as (
  select distinct on (pin, valid, emv_land, emv_bldg, emv_total, year_built, sale_date, sale_value, geom) *
  from parcels_union
)
select
  {{ dbt_utils.generate_surrogate_key(['ogc_fid', 'valid']) }} as parcel_id, *
from parcels_distinct
