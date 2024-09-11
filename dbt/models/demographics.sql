-- Demographic data
-- Contains data from the ACS and the computed segregation indexes.
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
),
demographics as (
  select * from acs_tract_with_description
  union all
  select * from segregation_indexes
)
-- Fill in data for 2011, 2012 using closest available year. Replace 2020 data
-- with 2019 data to avoid pandemic effects.
, demographics_replace_years as (
  select * from demographics where year_ != 2020
  union all
  select census_tract, 2020 as year_, name_, description, value_
  from demographics where year_ = 2019
  union all
  select census_tract, 2011 as year_, name_, description, value_
  from demographics where year_ = 2013
  union all
  select census_tract, 2012 as year_, name_, description, value_
  from demographics where year_ = 2013
)
select *
from demographics_replace_years
