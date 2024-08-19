with stops_2015 as (
  select
    2015 as year_
    , st_union(st_transform(geom, {{ var("srid") }}))::geometry(multipoint, {{ var("srid") }}) as geom
  from {{ source('minneapolis', 'high_frequency_transit_2015_freq_rail_stops') }}
)
, stops_2016 as ( -- stops are unchanged in 2016
  select
    2016 as year_
    , geom
  from stops_2015
)
select
  year_
  , geom
from stops_2015
union all
select
  year_
  , geom
from stops_2016
