select
  bdnum as ward_id
  , geom
from
  {{ source('minneapolis', 'wards_minneapolis') }}
