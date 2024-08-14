select
  geom
from
  {{ source('minneapolis', 'minneapolis_city_boundary') }}
