{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['date_', 'zcta_id', 'flow_direction', 'flow_type'], 'unique': true},
    ]
  )
}}

with
usps_migration as (select * from {{ ref('stg_usps_migration_add_zcta') }})
select
  date_,
  flow_direction,
  flow_type,
  zcta_id,
  sum(flow_value) as flow_value
from usps_migration
group by 1,2,3,4
