{% set usps_migration_flow_types = ['business', 'family', 'individual', 'perm', 'temp'] %}
{% set usps_migration_flow_directions = ['from', 'to'] %}

with process_date as (
  select to_date(yyyymm, 'YYYYMM') as date_, *
    from {{ source('minneapolis_old', 'usps_migration_raw') }}
)
, zip_codes as (
  select
    zip_code_id
    , zip_code
    , valid
  from
    {{ ref('zip_codes') }}
)
, add_zip_id as (
  select zip_code_id, process_date.*
  from
    process_date
    inner join zip_codes
        on zip_codes.zip_code = replace(process_date.zip_code, '=', '')
        and process_date.date_ <@ zip_codes.valid
)
{% for flow_direction in usps_migration_flow_directions %}
  select
    date_
    , zip_code_id
    , '{{ flow_direction }}' as flow_direction
    , 'total' as flow_type
    , total_{{ flow_direction }}_zip as flow_value
  from add_zip_id
  union all
  {% for flow_type in usps_migration_flow_types %}
    select
      date_
      , zip_code_id
      , '{{ flow_direction }}' as flow_direction
      , '{{ flow_type }}' as flow_type
      , total_{{ flow_direction }}_zip_{{ flow_type }} as flow_value
    from add_zip_id
  {% if not loop.last %} union all {% endif %}
  {% endfor %}
{% if not loop.last %} union all {% endif %}
{% endfor %}
