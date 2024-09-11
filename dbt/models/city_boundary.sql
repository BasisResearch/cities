select
  ogc_fid as city_boundary_id
  , st_transform(geom, {{ var("srid") }}) as geom
from
  {{ source('minneapolis', 'city_boundary_minneapolis') }}
