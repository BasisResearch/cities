select
  zcta5ce20 as zip_code,
  geom
from {{ source('minneapolis_old', 'zip_raw_2020') }}
