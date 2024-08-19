with lines_2015 as (
  select
    ogc_fid as line_id,
    st_transform(geom, {{ var("srid") }}) as geom
  from
    {{ source('minneapolis', 'high_frequency_transit_2015_freq_lines') }}
)
, lines_2016 as (
  select
    ogc_fid as line_id,
    st_transform(geom, {{ var("srid") }}) as geom
  from
    {{ source('minneapolis', 'high_frequency_transit_2016_freq_lines') }}
)
select
    2015 as year_,
    line_id,
    geom
from lines_2015
union all
select
    2016 as year_,
    line_id,
    geom
from lines_2016
