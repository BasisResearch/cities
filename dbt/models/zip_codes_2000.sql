select
  zcta as zip_code,
  ST_Union(geom) as geom
from
  {{ source('minneapolis_old', 'zip_raw_2000') }}
group by zcta
