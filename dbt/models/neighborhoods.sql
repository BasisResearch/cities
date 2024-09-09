select
  bdnum as neighborhood_id
  , bdname as name_
  , st_transform(geom, {{ var("srid") }}) as geom
from
  {{ source('minneapolis', 'neighborhoods_minneapolis') }}
