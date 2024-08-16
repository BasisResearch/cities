select
  bdnum as neighborhood_id
  , bdname as name_
  , geom
from
  {{ source('minneapolis', 'neighborhoods_minneapolis') }}
