with census_tracts as (
  select
    census_tract_id
    , statefp || countyfp || tractce as census_tract
  from {{ ref('census_tracts') }}
)
, parcels as (
  select
    parcel_id
    , geom
  from {{ ref('parcels') }}
)
, residential_permits as (
  select
    residential_permit_id
    , year_
    , permit_value
    , num_units
  from {{ ref('residential_permits') }}
)
, residential_permits_to_parcels as (
  select
    residential_permit_id
    , parcel_id
  from {{ ref('residential_permits_to_parcels') }}
)
, residential_permits_to_census_tracts as (
  select
    residential_permit_id
    , census_tract_id
  from {{ ref('residential_permits_to_census_tracts') }}
)
, residential as (
  select
    census_tracts.census_tract
    , residential_permits.year_
    , residential_permits.num_units
    , st_area(parcels.geom) as parcel_sqm
    , residential_permits.permit_value
  from
    residential_permits
    inner join residential_permits_to_parcels using (residential_permit_id)
    inner join parcels using (parcel_id)
    inner join residential_permits_to_census_tracts using (residential_permit_id)
    inner join census_tracts using (census_tract_id)
  where year_ <= 2020
)
, agg_residential as (
  select
    census_tract
    , year_
    , sum(num_units) as num_units
  from residential
  group by census_tract, year_
)

select
  census_tract
  , year_
  , num_units -- do we really want the total _applied_ units, or should we be
              -- looking at the total unit estimates from ACS?
from
  agg_residential
