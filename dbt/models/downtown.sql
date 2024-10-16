select
  ogc_fid as downtown_id
  , st_transform(geom, {{ var("srid") }}) as geom
from
  {{ source('minneapolis', 'downtown') }}
