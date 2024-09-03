select
  zcta5ce20 as zip_code,
  st_transform(geom, {{ var("srid") }}) as geom
from {{ source('minneapolis', 'zip_codes_tl_2020_us_zcta520') }}
