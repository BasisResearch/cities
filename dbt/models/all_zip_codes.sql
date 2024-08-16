with
zip_codes as (
select
  zip_code,
  '[2020-01-01,)'::daterange as valid,
  geom
from {{ ref('all_zip_codes_2020') }}
union all
select
  zip_code,
  '[,2020-01-01)'::daterange as valid,
  geom
from {{ ref('all_zip_codes_2010') }}
)
select
  {{ dbt_utils.generate_surrogate_key(['zip_code', 'valid']) }} as zip_code_id
  , zip_code
  , valid
  , geom
from zip_codes
