select
  bdnum as ward_id
  , geom
from
  {{ source('minneapolis', 'minneapolis_wards') }}
