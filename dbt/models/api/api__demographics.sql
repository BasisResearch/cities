{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['description']}
    ]
  )
}}

-- This is used by the web app. It has a row for each tract, demographic
-- variable pair and a column for each year.
with
demographics as (select * from {{ ref('demographics') }}),
census_tracts as (select * from {{ ref('census_tracts_in_city_boundary') }}),
demographics_filtered as (
  select demographics.*
  from demographics
  inner join census_tracts using (census_tract, year_)
),
final_ as (
  select
    description,
    census_tract as tract_id,
    {{ dbt_utils.pivot('year_',
                       dbt_utils.get_column_values(ref('demographics'),
                                                   'year_',
                                                   order_by='year_'),
                       then_value='value_',
                       else_value='null',
                       agg='max') }}
  from demographics_filtered
  group by 1, 2
)
select * from final_
