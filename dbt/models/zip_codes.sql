with
zip_codes as (
select
  zcta5ce20 as zip_code,
  '[2020-01-01,)'::daterange as valid,
  geom
from zip_raw_2020
union all
select
  zcta as zip_code,
  '[2000-01-01,2020-01-01)'::daterange as valid,
  ST_Transform(geom, 4269) as geom
from zip_raw_2000
)
select
  {{ dbt_utils.generate_surrogate_key(['zip_code', 'valid']) }} as zip_code_id
  , zip_code
  , valid
  , geom
from zip_codes
