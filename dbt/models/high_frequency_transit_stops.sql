with stops_2015 as (
  select
    st_union(st_transform(geom, {{ var("srid") }})) as geom
  from {{ source('minneapolis', 'high_frequency_transit_2015_freq_rail_stops') }}
)
select
  0 as high_frequency_transit_stop_id
  , '[,]'::daterange as valid
  , geom
from stops_2015
