select
  geom
from
  {{ source('minneapolis', 'city_boundary_minneapolis') }}
