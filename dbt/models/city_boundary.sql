select
  st_transform(geom, {{ var("srid") }}) as geom
from
  {{ source('minneapolis', 'city_boundary_minneapolis') }}
