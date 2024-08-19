with lines as (
  select
    line_id
    , year_
    , geom
  from {{ ref('high_frequency_transit_lines_union') }}
)
select
  {{ dbt_utils.generate_surrogate_key(['line_id', 'year_']) }} as line_id
  , year_
  , geom
from
  lines
