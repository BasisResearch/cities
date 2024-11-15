with
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }}),
residential_permits as (select * from {{ ref('residential_permits') }}),
residential_permits_to_census_tracts as (
  with
  residential_permits_tag as (
    select
      residential_permit_id as id
      , daterange(to_date(year_::text, 'YYYY'), to_date(year_::text, 'YYYY'), '[]') as valid
      , geom
    from residential_permits
  ),
  census_tracts_tag as (
    select census_tract_id as id, valid, geom from census_tracts
  )
  select
    child_id as residential_permit_id,
    parent_id as census_tract_id,
    valid,
    type_
  from {{ tag_regions("residential_permits_tag", "census_tracts_tag") }}
)
select
  census_tracts.census_tract_id,
  sum(residential_permits.num_units)::int as num_units
from
  census_tracts
  left join residential_permits_to_census_tracts using (census_tract_id)
  left join residential_permits using (residential_permit_id)
group by 1
