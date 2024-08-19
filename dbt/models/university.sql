select
  st_transform(geom, {{ var("srid") }}) as geom
from
  {{ source('minneapolis', 'university') }}
