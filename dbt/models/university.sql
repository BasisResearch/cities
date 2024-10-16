select
  ogc_fid as university_id
  , st_transform(geom, {{ var("srid") }}) as geom
from
  {{ source('minneapolis', 'university') }}
