{% set usps_migration_flow_types = ['business', 'family', 'individual', 'perm', 'temp'] %}
{% set usps_migration_flow_directions = ['from', 'to'] %}

with
process_date as (
  select to_date(yyyy_mm, 'YYYYMM') as date_, *
  from {{ ref('stg_usps_migration_union') }}
)
{% for flow_direction in usps_migration_flow_directions %}
  select
    date_
    , zip_code
    , '{{ flow_direction }}' as flow_direction
    , 'total' as flow_type
    , total_{{ flow_direction }}_zip::int as flow_value
  from process_date
  union all
  {% for flow_type in usps_migration_flow_types %}
    select
      date_
      , zip_code
      , '{{ flow_direction }}' as flow_direction
      , '{{ flow_type }}' as flow_type
      , total_{{ flow_direction }}_zip_{{ flow_type }}::int as flow_value
    from process_date
  {% if not loop.last %} union all {% endif %}
  {% endfor %}
{% if not loop.last %} union all {% endif %}
{% endfor %}
