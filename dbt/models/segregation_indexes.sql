{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract', 'year_', 'distribution'], 'unique': true},
    ]
  )
}}

-- Segregation index for each tract for each year, computed for each reference
-- distribution.
--
-- The segregation index is the KL-divergence between the distribution of
-- population in a tract and a reference distribution. For example, a tract that
-- has many more white people than the average for the city will have a high
-- segregation index for the 'average_city' distribution.
with
  categories as (select * from {{ ref("population_categories") }})
  , acs_tract_all as (select * from {{ ref("acs_tract") }})
  , acs_variables as (
    select
      variable as name_,
      description
    from {{ ref("acs_variables") }}
  )
  , census_tracts_in_city_boundary as (select * from {{ ref('census_tracts_in_city_boundary') }})
  , acs_tract as (
    select * from acs_tract_all inner join census_tracts_in_city_boundary using (census_tract, year_)
    )
  , pop_tyc as
    ( -- Population by tract, year, and category
    select acs_tract.census_tract, acs_tract.year_, categories.category, acs_tract.value_
      from acs_tract
      inner join acs_variables using (name_)
      inner join categories on categories.category = acs_variables.description
    ),
  pop_ty as
    (
    select census_tract, year_, sum(value_) as value_
      from pop_tyc
     group by 1, 2
    ),
  pop_yc as
    ( -- Population by year and category
    select year_, category, sum(value_) as value_
      from pop_tyc
     group by 1, 2
    ),
  pop_y as
    ( -- Population by year
    select year_, sum(value_) as value_
      from pop_tyc
     group by 1
    ),
  dist_yc as
    ( -- Distribution of population by year and category
    select
      pop_yc.year_,
      pop_yc.category,
      ({{ safe_divide('pop_yc.value_', 'pop_y.value_') }})::double precision as value_
    from pop_yc inner join pop_y using (year_)
    ),
  dist_tyc as
    (  -- Distribution of population by tract, year, and category
    select
      pop_tyc.census_tract,
      pop_tyc.year_,
      pop_tyc.category,
      ({{ safe_divide('pop_tyc.value_', 'pop_ty.value_') }})::double precision as value_
    from pop_tyc inner join pop_ty using (year_, census_tract)
    ),
  uniform_dist as
    ( -- Uniform distribution across categories
    with n_cat as (select count(*) as n_cat from categories)
    select category, (1.0 / n_cat)::double precision as value_
      from categories, n_cat
    ),
  average_dist as
    ( -- Average of the annual citywide distributions
    select category, avg(value_)::double precision as value_
      from dist_yc
    group by 1
    )
select
  census_tract,
  year_,
  dist as distribution,
  sum(case when p = 0 or q = 0 then 0 else p * ln(p / q) end) as segregation_index
from
  (
    select
      dist_tyc.census_tract,
      dist_tyc.year_,
      dist_tyc.value_ as p,
      uniform_dist.value_ as q,
      'uniform' as dist
    from dist_tyc inner join uniform_dist using (category)
    union all
    select
      dist_tyc.census_tract,
      dist_tyc.year_,
      dist_tyc.value_ as p,
      dist_yc.value_ as q,
      'annual_city' as dist
    from dist_tyc inner join dist_yc using (year_, category)
    union all
    select
      dist_tyc.census_tract,
      dist_tyc.year_,
      dist_tyc.value_ as p,
      average_dist.value_ as q,
      'average_city' as dist
    from dist_tyc inner join average_dist using (category)
  )
group by 1, 2, 3
