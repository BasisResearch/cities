select
  zcta as zip_code,
  ST_Union(geom) as geom
from
  zip_raw_2000
group by zcta
