with
acs_tract as (select * from {{ ref('acs_tract') }}),
acs_variables as (select * from {{ ref('acs_variables') }}),
acs_tract_with_description as (
  select
    acs_tract.census_tract,
    acs_tract.year_,
    acs_tract.name_,
    acs_variables.description,
    acs_tract.value_
  from acs_tract
  inner join acs_variables on acs_tract.name_ = acs_variables.variable
),
segregation_indexes as (
  select
    census_tract,
    year_,
    null as name_,
    'segregation_index_' || distribution as description,
    segregation_index as value_
  from {{ ref('segregation_indexes') }}
)
select * from acs_tract_with_description
union all
select * from segregation_indexes
