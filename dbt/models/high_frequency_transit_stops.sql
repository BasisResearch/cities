with stops_2015 as (
  select
    st_union(st_transform(geom, {{ var("srid") }}))::geometry(multipoint, {{ var("srid") }}) as geom
  from {{ source('minneapolis', 'high_frequency_transit_2015_freq_rail_stops') }}
)
select
  '[,]'::daterange as valid
  , geom
from stops_2015
