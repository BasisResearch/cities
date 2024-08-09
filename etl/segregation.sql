create or replace view categories as select * from (
  values
  ('population_white_non_hispanic'),
  ('population_black_non_hispanic'),
  ('population_hispanic_or_latino'),
  ('population_asian_non_hispanic'),
  ('population_native_hawaiian_or_pacific_islander_non_hispanic'),
  ('population_american_indian_or_alaska_native_non_hispanic'),
  ('population_multiple_races_non_hispanic'),
  ('population_other_non_hispanic')
) as t (description);

drop type if exists reference_distribution cascade;
create type reference_distribution as enum (
    'uniform'
    , 'annual_city'
    , 'average_city'
);


-- Segregation index for each tract for each year, computed for each reference
-- distribution.
--
-- The segregation index is the KL-divergence between the distribution of
-- population in a tract and a reference distribution. For example, a tract that
-- has many more white people than the average for the city will have a high
-- segregation index for the 'average_city' distribution.

drop table if exists segregation;

create table segregation as (
with
  pop_tyc as
    ( -- Population by tract, year, and category
    select id, year_, description, value_
      from acs_tract
           join acs_variable using (name_)
           join categories using (description)
    ),
  pop_ty as
    ( -- Population by tract and year (note: using 'population' variable instead of aggregating categories)
    select id, year_, value_
      from acs_tract join acs_variable using (name_)
     where description = 'population'
    ),
  pop_yc as
    ( -- Population by year and category
    select year_, description, sum(value_) as value_
      from pop_tyc group by year_, description
    ),
  pop_y as
    ( -- Population by year
    select year_, sum(value_) as value_ from pop_ty group by year_
    ),
  dist_yc as
    ( -- Distribution of population by year and category
    select description, c.year_,
           case t.value_ when 0 then 0 else c.value_ / t.value_ end as value_
      from pop_yc as c join pop_y as t using (year_)
    ),
  dist_tyc as
    (  -- Distribution of population by tract, year, and category
    select id, year_, description,
           case t.value_ when 0 then 0 else p.value_ / t.value_ end as value_
      from pop_tyc as p join pop_ty as t using (year_, id)
    ),
  uniform_dist as
    ( -- Uniform distribution across categories
    with n_cat as (select count(*) as n_cat from categories)
    select description, 1.0 / n_cat as value_
      from categories, n_cat
    ),
  average_dist as
    ( -- Average of the annual citywide distributions
    select description, avg(value_) as value_
      from dist_yc
     group by description
    )
select id, year_, dist, sum(case when p = 0 or q = 0 then 0 else p * ln(p / q) end) as segregation_index
  from
    (
      select id, year_, 'uniform'::reference_distribution as dist, dist_tyc.value_ as p, uniform_dist.value_ as q
        from dist_tyc join uniform_dist using (description)
       union all
      select id, year_, 'annual_city'::reference_distribution as dist, dist_tyc.value_ as p, dist_yc.value_ as q
        from dist_tyc join dist_yc using (year_, description)
       union all
      select id, year_, 'average_city'::reference_distribution as dist, dist_tyc.value_ as p, average_dist.value_ as q
        from dist_tyc join average_dist using (description)
    )
 group by id, year_, dist
);
