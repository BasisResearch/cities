-- Demographic data
-- Contains data from the ACS and the computed segregation indexes.
with
demographics as (select * from {{ ref('stg_demographics_union') }}),
-- Fill in data for 2011, 2012 using closest available year. Replace 2020 data
-- with 2019 data to avoid pandemic effects.
demographics_replace_years as (
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
